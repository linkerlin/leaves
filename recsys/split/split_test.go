package split

import (
	"strings"
	"testing"
	"time"

	"github.com/linkerlin/leaves/v2/recsys/contract"
)

func at(day int) time.Time {
	return time.Date(2026, 8, day, 0, 0, 0, 0, time.UTC)
}

func ev(id string, day int, typ contract.EventType) contract.InteractionEvent {
	return contract.InteractionEvent{
		EventID: id, OccurredAt: at(day), SubjectID: "u1", ItemID: "i" + id, EventType: typ, Source: "test",
	}
}

func cfg() TimeConfig {
	return TimeConfig{TrainEnd: at(10), ValStart: at(12), TestStart: at(20)}
}

func TestSplitPartitionsAndDropsGap(t *testing.T) {
	events := []contract.InteractionEvent{
		ev("a", 1, contract.EventClick), ev("b", 9, contract.EventClick),
		ev("gap", 10, contract.EventClick), // 落在隔离带 [TrainEnd, ValStart)
		ev("c", 13, contract.EventClick),
		ev("d", 25, contract.EventClick),
	}
	train, val, test, err := Split(events, cfg())
	if err != nil {
		t.Fatal(err)
	}
	if len(train) != 2 || len(val) != 1 || len(test) != 1 {
		t.Fatalf("partition: train=%d val=%d test=%d", len(train), len(val), len(test))
	}
	if train[0].EventID != "a" || val[0].EventID != "c" || test[0].EventID != "d" {
		t.Fatalf("wrong events: %+v %+v %+v", train, val, test)
	}
	if err := CheckLeakage(train, cfg().ValStart); err != nil {
		t.Fatal(err)
	}
}

func TestSplitRejectsBadOrder(t *testing.T) {
	bad := TimeConfig{TrainEnd: at(12), ValStart: at(10), TestStart: at(20)}
	if _, _, _, err := Split(nil, bad); err == nil || !strings.Contains(err.Error(), "train_end < validation_start") {
		t.Fatalf("want order failure, got %v", err)
	}
	local := TimeConfig{TrainEnd: time.Date(2026, 8, 10, 0, 0, 0, 0, time.FixedZone("X", 8*3600)), ValStart: at(12), TestStart: at(20)}
	if _, _, _, err := Split(nil, local); err == nil || !strings.Contains(err.Error(), "UTC") {
		t.Fatalf("want non-UTC failure, got %v", err)
	}
}

func TestCheckLeakageCatchesFutureEvent(t *testing.T) {
	train := []contract.InteractionEvent{ev("future", 15, contract.EventClick)}
	if err := CheckLeakage(train, at(12)); err == nil || !strings.Contains(err.Error(), "leakage") {
		t.Fatalf("want leakage failure, got %v", err)
	}
}

func TestColdStartUsers(t *testing.T) {
	train := []contract.InteractionEvent{ev("a", 1, contract.EventClick)}
	eval := []contract.InteractionEvent{
		{EventID: "b", OccurredAt: at(13), SubjectID: "u1", ItemID: "i1", EventType: contract.EventClick},
		{EventID: "c", OccurredAt: at(14), SubjectID: "new", ItemID: "i2", EventType: contract.EventClick},
	}
	cold := ColdStartUsers(train, eval)
	if !cold["new"] || cold["u1"] {
		t.Fatalf("cold set wrong: %+v", cold)
	}
}

func TestAssignDeterministicOrder(t *testing.T) {
	events := []contract.InteractionEvent{
		{EventID: "z", OccurredAt: at(5), SubjectID: "u1", ItemID: "i1", EventType: contract.EventClick, Value: 1},
		{EventID: "a", OccurredAt: at(5), SubjectID: "u2", ItemID: "i2", EventType: contract.EventRating, Value: 4},
		{EventID: "m", OccurredAt: at(3), SubjectID: "u3", ItemID: "i3", EventType: contract.EventClick, Value: 2},
	}
	got := Assign(events)
	// (occurred_at, event_id) 排序：m(3日) 先，再 a/z 同日按 id a<z
	if got[0].User != "u3" || got[1].User != "u2" || got[2].User != "u1" {
		t.Fatalf("order wrong: %+v", got)
	}
}

func TestTimeRangeOfEvents(t *testing.T) {
	r := TimeRange([]contract.InteractionEvent{ev("a", 3, contract.EventClick), ev("b", 7, contract.EventClick)})
	if !r.Start.Equal(at(3)) || !r.End.After(at(7)) {
		t.Fatalf("range wrong: %+v", r)
	}
}
