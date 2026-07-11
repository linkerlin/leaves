package io

// 极小 protobuf wire 读写，仅服务 ONNX TreeEnsemble 子集（LIB-10）。
// 不依赖 google.golang.org/protobuf / 完整 ONNX 运行时。

import (
	"encoding/binary"
	"fmt"
	"math"
)

const (
	wireVarint = 0
	wireBytes  = 2
	wire32     = 5
)

// ---- writer ----

func pbAppendVarint(b []byte, x uint64) []byte {
	for x >= 0x80 {
		b = append(b, byte(x)|0x80)
		x >>= 7
	}
	return append(b, byte(x))
}

func pbAppendTag(b []byte, field, wire int) []byte {
	return pbAppendVarint(b, uint64(field<<3|wire))
}

func pbAppendBytes(b []byte, field int, data []byte) []byte {
	b = pbAppendTag(b, field, wireBytes)
	b = pbAppendVarint(b, uint64(len(data)))
	return append(b, data...)
}

func pbAppendString(b []byte, field int, s string) []byte {
	return pbAppendBytes(b, field, []byte(s))
}

func pbAppendInt64(b []byte, field int, v int64) []byte {
	b = pbAppendTag(b, field, wireVarint)
	return pbAppendVarint(b, uint64(v))
}

func pbAppendFloat32(b []byte, field int, v float32) []byte {
	b = pbAppendTag(b, field, wire32)
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
	return append(b, buf[:]...)
}

// ---- reader ----

type pbReader struct {
	b []byte
	i int
}

func (r *pbReader) remaining() int { return len(r.b) - r.i }

func (r *pbReader) readVarint() (uint64, error) {
	var x uint64
	var s uint
	for {
		if r.i >= len(r.b) {
			return 0, fmt.Errorf("protobuf: truncated varint")
		}
		c := r.b[r.i]
		r.i++
		if c < 0x80 {
			return x | uint64(c)<<s, nil
		}
		x |= uint64(c&0x7f) << s
		s += 7
		if s > 63 {
			return 0, fmt.Errorf("protobuf: varint overflow")
		}
	}
}

func (r *pbReader) readTag() (field, wire int, err error) {
	v, err := r.readVarint()
	if err != nil {
		return 0, 0, err
	}
	return int(v >> 3), int(v & 7), nil
}

func (r *pbReader) skip(wire int) error {
	switch wire {
	case wireVarint:
		_, err := r.readVarint()
		return err
	case wireBytes:
		n, err := r.readVarint()
		if err != nil {
			return err
		}
		if r.remaining() < int(n) {
			return fmt.Errorf("protobuf: truncated bytes")
		}
		r.i += int(n)
		return nil
	case wire32:
		if r.remaining() < 4 {
			return fmt.Errorf("protobuf: truncated fixed32")
		}
		r.i += 4
		return nil
	case 1: // 64-bit
		if r.remaining() < 8 {
			return fmt.Errorf("protobuf: truncated fixed64")
		}
		r.i += 8
		return nil
	default:
		return fmt.Errorf("protobuf: unknown wire %d", wire)
	}
}

func (r *pbReader) readBytes() ([]byte, error) {
	n, err := r.readVarint()
	if err != nil {
		return nil, err
	}
	if r.remaining() < int(n) {
		return nil, fmt.Errorf("protobuf: truncated bytes payload")
	}
	out := r.b[r.i : r.i+int(n)]
	r.i += int(n)
	return out, nil
}

func (r *pbReader) readString() (string, error) {
	b, err := r.readBytes()
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func (r *pbReader) readFloat32() (float32, error) {
	if r.remaining() < 4 {
		return 0, fmt.Errorf("protobuf: truncated float")
	}
	u := binary.LittleEndian.Uint32(r.b[r.i : r.i+4])
	r.i += 4
	return math.Float32frombits(u), nil
}

// parseAttribute 解析 ONNX AttributeProto 子集。
type onnxAttr struct {
	name    string
	i       int64
	f       float32
	floats  []float64
	ints    []int64
	strings []string
}

func parseONNXAttribute(data []byte) (onnxAttr, error) {
	var a onnxAttr
	r := pbReader{b: data}
	for r.remaining() > 0 {
		field, wire, err := r.readTag()
		if err != nil {
			return a, err
		}
		switch {
		case field == 1 && wire == wireBytes: // name
			a.name, err = r.readString()
		case field == 8 && wire == wireVarint: // i
			var v uint64
			v, err = r.readVarint()
			a.i = int64(v)
		case field == 9 && wire == wire32: // f
			var f float32
			f, err = r.readFloat32()
			a.f = f
		case field == 10 && wire == wire32: // floats (unpacked)
			var f float32
			f, err = r.readFloat32()
			a.floats = append(a.floats, float64(f))
		case field == 10 && wire == wireBytes: // floats packed
			var raw []byte
			raw, err = r.readBytes()
			if err == nil {
				for j := 0; j+4 <= len(raw); j += 4 {
					u := binary.LittleEndian.Uint32(raw[j : j+4])
					a.floats = append(a.floats, float64(math.Float32frombits(u)))
				}
			}
		case field == 11 && wire == wireVarint: // ints unpacked
			var v uint64
			v, err = r.readVarint()
			a.ints = append(a.ints, int64(v))
		case field == 11 && wire == wireBytes: // ints packed
			var raw []byte
			raw, err = r.readBytes()
			if err == nil {
				rr := pbReader{b: raw}
				for rr.remaining() > 0 {
					v, e2 := rr.readVarint()
					if e2 != nil {
						err = e2
						break
					}
					a.ints = append(a.ints, int64(v))
				}
			}
		case field == 12 && wire == wireBytes: // strings
			var s string
			s, err = r.readString()
			a.strings = append(a.strings, s)
		default:
			err = r.skip(wire)
		}
		if err != nil {
			return a, err
		}
	}
	return a, nil
}

type onnxNode struct {
	opType string
	domain string
	attrs  map[string]onnxAttr
}

func parseONNXNode(data []byte) (onnxNode, error) {
	n := onnxNode{attrs: map[string]onnxAttr{}}
	r := pbReader{b: data}
	for r.remaining() > 0 {
		field, wire, err := r.readTag()
		if err != nil {
			return n, err
		}
		switch {
		case field == 4 && wire == wireBytes: // op_type
			n.opType, err = r.readString()
		case field == 5 && wire == wireBytes: // domain
			n.domain, err = r.readString()
		case field == 6 && wire == wireBytes: // attribute
			var raw []byte
			raw, err = r.readBytes()
			if err == nil {
				var a onnxAttr
				a, err = parseONNXAttribute(raw)
				if err == nil {
					n.attrs[a.name] = a
				}
			}
		default:
			err = r.skip(wire)
		}
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

func parseONNXGraph(data []byte) ([]onnxNode, error) {
	var nodes []onnxNode
	r := pbReader{b: data}
	for r.remaining() > 0 {
		field, wire, err := r.readTag()
		if err != nil {
			return nil, err
		}
		if field == 1 && wire == wireBytes { // node
			raw, err := r.readBytes()
			if err != nil {
				return nil, err
			}
			n, err := parseONNXNode(raw)
			if err != nil {
				return nil, err
			}
			nodes = append(nodes, n)
			continue
		}
		if err := r.skip(wire); err != nil {
			return nil, err
		}
	}
	return nodes, nil
}

func parseONNXModel(data []byte) ([]onnxNode, error) {
	r := pbReader{b: data}
	for r.remaining() > 0 {
		field, wire, err := r.readTag()
		if err != nil {
			return nil, err
		}
		if field == 7 && wire == wireBytes { // graph
			raw, err := r.readBytes()
			if err != nil {
				return nil, err
			}
			return parseONNXGraph(raw)
		}
		if err := r.skip(wire); err != nil {
			return nil, err
		}
	}
	return nil, fmt.Errorf("onnx: no graph in model")
}
