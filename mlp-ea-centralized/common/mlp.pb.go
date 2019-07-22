// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mlp.proto

package common

import (
	encoding_binary "encoding/binary"
	fmt "fmt"
	_ "github.com/gogo/protobuf/gogoproto"
	proto "github.com/gogo/protobuf/proto"
	io "io"
	math "math"
	math_bits "math/bits"
	reflect "reflect"
	strings "strings"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.GoGoProtoPackageIsVersion2 // please upgrade the proto package

type TransferFunc int32

const (
	TransferFunc_SIGMOIDAL TransferFunc = 0
)

var TransferFunc_name = map[int32]string{
	0: "SIGMOIDAL",
}

var TransferFunc_value = map[string]int32{
	"SIGMOIDAL": 0,
}

func (x TransferFunc) String() string {
	return proto.EnumName(TransferFunc_name, int32(x))
}

func (TransferFunc) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_524ae208a70ac1d9, []int{0}
}

type NeuronUnit struct {
	Weights []float64 `protobuf:"fixed64,1,rep,packed,name=Weights,proto3" json:"Weights,omitempty"`
	Bias    float64   `protobuf:"fixed64,2,opt,name=Bias,proto3" json:"Bias,omitempty"`
	Lrate   float64   `protobuf:"fixed64,3,opt,name=Lrate,proto3" json:"Lrate,omitempty"`
	Value   float64   `protobuf:"fixed64,4,opt,name=Value,proto3" json:"Value,omitempty"`
	Delta   float64   `protobuf:"fixed64,5,opt,name=Delta,proto3" json:"Delta,omitempty"`
}

func (m *NeuronUnit) Reset()      { *m = NeuronUnit{} }
func (*NeuronUnit) ProtoMessage() {}
func (*NeuronUnit) Descriptor() ([]byte, []int) {
	return fileDescriptor_524ae208a70ac1d9, []int{0}
}
func (m *NeuronUnit) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *NeuronUnit) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_NeuronUnit.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *NeuronUnit) XXX_Merge(src proto.Message) {
	xxx_messageInfo_NeuronUnit.Merge(m, src)
}
func (m *NeuronUnit) XXX_Size() int {
	return m.Size()
}
func (m *NeuronUnit) XXX_DiscardUnknown() {
	xxx_messageInfo_NeuronUnit.DiscardUnknown(m)
}

var xxx_messageInfo_NeuronUnit proto.InternalMessageInfo

func (m *NeuronUnit) GetWeights() []float64 {
	if m != nil {
		return m.Weights
	}
	return nil
}

func (m *NeuronUnit) GetBias() float64 {
	if m != nil {
		return m.Bias
	}
	return 0
}

func (m *NeuronUnit) GetLrate() float64 {
	if m != nil {
		return m.Lrate
	}
	return 0
}

func (m *NeuronUnit) GetValue() float64 {
	if m != nil {
		return m.Value
	}
	return 0
}

func (m *NeuronUnit) GetDelta() float64 {
	if m != nil {
		return m.Delta
	}
	return 0
}

type NeuralLayer struct {
	NeuronUnits []NeuronUnit `protobuf:"bytes,1,rep,name=NeuronUnits,proto3" json:"NeuronUnits"`
	Length      int64        `protobuf:"varint,2,opt,name=Length,proto3" json:"Length,omitempty"`
}

func (m *NeuralLayer) Reset()      { *m = NeuralLayer{} }
func (*NeuralLayer) ProtoMessage() {}
func (*NeuralLayer) Descriptor() ([]byte, []int) {
	return fileDescriptor_524ae208a70ac1d9, []int{1}
}
func (m *NeuralLayer) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *NeuralLayer) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_NeuralLayer.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *NeuralLayer) XXX_Merge(src proto.Message) {
	xxx_messageInfo_NeuralLayer.Merge(m, src)
}
func (m *NeuralLayer) XXX_Size() int {
	return m.Size()
}
func (m *NeuralLayer) XXX_DiscardUnknown() {
	xxx_messageInfo_NeuralLayer.DiscardUnknown(m)
}

var xxx_messageInfo_NeuralLayer proto.InternalMessageInfo

func (m *NeuralLayer) GetNeuronUnits() []NeuronUnit {
	if m != nil {
		return m.NeuronUnits
	}
	return nil
}

func (m *NeuralLayer) GetLength() int64 {
	if m != nil {
		return m.Length
	}
	return 0
}

type MultiLayerNetwork struct {
	LRate        float64       `protobuf:"fixed64,1,opt,name=L_rate,json=LRate,proto3" json:"L_rate,omitempty"`
	NeuralLayers []NeuralLayer `protobuf:"bytes,2,rep,name=NeuralLayers,proto3" json:"NeuralLayers"`
	TFunc        TransferFunc  `protobuf:"varint,3,opt,name=T_func,json=TFunc,proto3,enum=common.TransferFunc" json:"T_func,omitempty"`
}

func (m *MultiLayerNetwork) Reset()      { *m = MultiLayerNetwork{} }
func (*MultiLayerNetwork) ProtoMessage() {}
func (*MultiLayerNetwork) Descriptor() ([]byte, []int) {
	return fileDescriptor_524ae208a70ac1d9, []int{2}
}
func (m *MultiLayerNetwork) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *MultiLayerNetwork) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_MultiLayerNetwork.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *MultiLayerNetwork) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MultiLayerNetwork.Merge(m, src)
}
func (m *MultiLayerNetwork) XXX_Size() int {
	return m.Size()
}
func (m *MultiLayerNetwork) XXX_DiscardUnknown() {
	xxx_messageInfo_MultiLayerNetwork.DiscardUnknown(m)
}

var xxx_messageInfo_MultiLayerNetwork proto.InternalMessageInfo

func (m *MultiLayerNetwork) GetLRate() float64 {
	if m != nil {
		return m.LRate
	}
	return 0
}

func (m *MultiLayerNetwork) GetNeuralLayers() []NeuralLayer {
	if m != nil {
		return m.NeuralLayers
	}
	return nil
}

func (m *MultiLayerNetwork) GetTFunc() TransferFunc {
	if m != nil {
		return m.TFunc
	}
	return TransferFunc_SIGMOIDAL
}

func init() {
	proto.RegisterEnum("common.TransferFunc", TransferFunc_name, TransferFunc_value)
	proto.RegisterType((*NeuronUnit)(nil), "common.NeuronUnit")
	proto.RegisterType((*NeuralLayer)(nil), "common.NeuralLayer")
	proto.RegisterType((*MultiLayerNetwork)(nil), "common.MultiLayerNetwork")
}

func init() { proto.RegisterFile("mlp.proto", fileDescriptor_524ae208a70ac1d9) }

var fileDescriptor_524ae208a70ac1d9 = []byte{
	// 382 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x54, 0x91, 0xc1, 0x4a, 0xeb, 0x40,
	0x14, 0x86, 0x33, 0xb7, 0x6d, 0x2e, 0x9d, 0xf6, 0x5e, 0xee, 0x1d, 0xab, 0x04, 0xc1, 0xb1, 0x74,
	0x55, 0x14, 0x53, 0xa9, 0x3b, 0xc1, 0x85, 0xa5, 0x28, 0x85, 0xb4, 0x42, 0xac, 0xba, 0x2c, 0xd3,
	0x30, 0x4d, 0x83, 0x49, 0xa6, 0xa4, 0x13, 0x44, 0x70, 0xe1, 0x23, 0xb8, 0xf2, 0x19, 0x7c, 0x04,
	0x1f, 0xa1, 0xcb, 0x2e, 0xbb, 0x12, 0x9b, 0xbe, 0x80, 0x4b, 0x97, 0x32, 0x33, 0x29, 0xc6, 0x55,
	0xe6, 0xfb, 0xcf, 0x1c, 0xce, 0x77, 0x32, 0xb0, 0x18, 0xf8, 0x13, 0x73, 0x12, 0x31, 0xce, 0x90,
	0xee, 0xb0, 0x20, 0x60, 0xe1, 0xf6, 0x81, 0xeb, 0xf1, 0x71, 0x3c, 0x34, 0x1d, 0x16, 0x34, 0x5c,
	0xe6, 0xb2, 0x86, 0x2c, 0x0f, 0xe3, 0x91, 0x24, 0x09, 0xf2, 0xa4, 0xda, 0x6a, 0x0f, 0x10, 0xf6,
	0x68, 0x1c, 0xb1, 0xf0, 0x2a, 0xf4, 0x38, 0x32, 0xe0, 0xef, 0x1b, 0xea, 0xb9, 0x63, 0x3e, 0x35,
	0x40, 0x35, 0x57, 0x07, 0xf6, 0x1a, 0x11, 0x82, 0xf9, 0x96, 0x47, 0xa6, 0xc6, 0xaf, 0x2a, 0xa8,
	0x03, 0x5b, 0x9e, 0x51, 0x05, 0x16, 0xac, 0x88, 0x70, 0x6a, 0xe4, 0x64, 0xa8, 0x40, 0xa4, 0xd7,
	0xc4, 0x8f, 0xa9, 0x91, 0x57, 0xa9, 0x04, 0x91, 0xb6, 0xa9, 0xcf, 0x89, 0x51, 0x50, 0xa9, 0x84,
	0x1a, 0x81, 0x25, 0x31, 0x9d, 0xf8, 0x16, 0xb9, 0xa7, 0x11, 0x3a, 0x56, 0xa8, 0x64, 0x94, 0x42,
	0xa9, 0x89, 0x4c, 0xb5, 0x99, 0xf9, 0x5d, 0x6a, 0xe5, 0x67, 0x6f, 0xbb, 0x9a, 0x9d, 0xbd, 0x8c,
	0xb6, 0xa0, 0x6e, 0xd1, 0xd0, 0xe5, 0x63, 0xa9, 0x98, 0xb3, 0x53, 0xaa, 0x3d, 0x03, 0xf8, 0xbf,
	0x1b, 0xfb, 0xdc, 0x93, 0x23, 0x7a, 0x94, 0xdf, 0xb1, 0xe8, 0x16, 0x6d, 0x42, 0xdd, 0x1a, 0x48,
	0x77, 0x90, 0xba, 0xdb, 0xc2, 0xfd, 0x04, 0x96, 0x33, 0x3e, 0x62, 0x5b, 0x61, 0xb0, 0x91, 0x35,
	0x48, 0x6b, 0xa9, 0xc2, 0x8f, 0xeb, 0x68, 0x1f, 0xea, 0xfd, 0xc1, 0x28, 0x0e, 0x1d, 0xf9, 0x47,
	0xfe, 0x36, 0x2b, 0xeb, 0xc6, 0x7e, 0x44, 0xc2, 0xe9, 0x88, 0x46, 0x67, 0x71, 0xe8, 0xd8, 0x85,
	0xbe, 0xf8, 0xec, 0xed, 0xc0, 0x72, 0x36, 0x46, 0x7f, 0x60, 0xf1, 0xb2, 0x73, 0xde, 0xbd, 0xe8,
	0xb4, 0x4f, 0xad, 0x7f, 0x5a, 0xeb, 0x70, 0xb1, 0xc4, 0xda, 0xc7, 0x12, 0x83, 0xcf, 0x25, 0x06,
	0x8f, 0x09, 0x06, 0x2f, 0x09, 0x06, 0xaf, 0x09, 0x06, 0xb3, 0x04, 0x83, 0x79, 0x82, 0xc1, 0x7b,
	0x82, 0xc1, 0xd3, 0x0a, 0x6b, 0xf3, 0x15, 0xd6, 0x16, 0x2b, 0xac, 0x0d, 0x75, 0xf9, 0xa2, 0x47,
	0x5f, 0x01, 0x00, 0x00, 0xff, 0xff, 0x0d, 0xa6, 0x57, 0x11, 0x15, 0x02, 0x00, 0x00,
}

func (this *NeuronUnit) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*NeuronUnit)
	if !ok {
		that2, ok := that.(NeuronUnit)
		if ok {
			that1 = &that2
		} else {
			return false
		}
	}
	if that1 == nil {
		return this == nil
	} else if this == nil {
		return false
	}
	if len(this.Weights) != len(that1.Weights) {
		return false
	}
	for i := range this.Weights {
		if this.Weights[i] != that1.Weights[i] {
			return false
		}
	}
	if this.Bias != that1.Bias {
		return false
	}
	if this.Lrate != that1.Lrate {
		return false
	}
	if this.Value != that1.Value {
		return false
	}
	if this.Delta != that1.Delta {
		return false
	}
	return true
}
func (this *NeuralLayer) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*NeuralLayer)
	if !ok {
		that2, ok := that.(NeuralLayer)
		if ok {
			that1 = &that2
		} else {
			return false
		}
	}
	if that1 == nil {
		return this == nil
	} else if this == nil {
		return false
	}
	if len(this.NeuronUnits) != len(that1.NeuronUnits) {
		return false
	}
	for i := range this.NeuronUnits {
		if !this.NeuronUnits[i].Equal(&that1.NeuronUnits[i]) {
			return false
		}
	}
	if this.Length != that1.Length {
		return false
	}
	return true
}
func (this *MultiLayerNetwork) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*MultiLayerNetwork)
	if !ok {
		that2, ok := that.(MultiLayerNetwork)
		if ok {
			that1 = &that2
		} else {
			return false
		}
	}
	if that1 == nil {
		return this == nil
	} else if this == nil {
		return false
	}
	if this.LRate != that1.LRate {
		return false
	}
	if len(this.NeuralLayers) != len(that1.NeuralLayers) {
		return false
	}
	for i := range this.NeuralLayers {
		if !this.NeuralLayers[i].Equal(&that1.NeuralLayers[i]) {
			return false
		}
	}
	if this.TFunc != that1.TFunc {
		return false
	}
	return true
}
func (this *NeuronUnit) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 9)
	s = append(s, "&common.NeuronUnit{")
	s = append(s, "Weights: "+fmt.Sprintf("%#v", this.Weights)+",\n")
	s = append(s, "Bias: "+fmt.Sprintf("%#v", this.Bias)+",\n")
	s = append(s, "Lrate: "+fmt.Sprintf("%#v", this.Lrate)+",\n")
	s = append(s, "Value: "+fmt.Sprintf("%#v", this.Value)+",\n")
	s = append(s, "Delta: "+fmt.Sprintf("%#v", this.Delta)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *NeuralLayer) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&common.NeuralLayer{")
	if this.NeuronUnits != nil {
		vs := make([]*NeuronUnit, len(this.NeuronUnits))
		for i := range vs {
			vs[i] = &this.NeuronUnits[i]
		}
		s = append(s, "NeuronUnits: "+fmt.Sprintf("%#v", vs)+",\n")
	}
	s = append(s, "Length: "+fmt.Sprintf("%#v", this.Length)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *MultiLayerNetwork) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&common.MultiLayerNetwork{")
	s = append(s, "LRate: "+fmt.Sprintf("%#v", this.LRate)+",\n")
	if this.NeuralLayers != nil {
		vs := make([]*NeuralLayer, len(this.NeuralLayers))
		for i := range vs {
			vs[i] = &this.NeuralLayers[i]
		}
		s = append(s, "NeuralLayers: "+fmt.Sprintf("%#v", vs)+",\n")
	}
	s = append(s, "TFunc: "+fmt.Sprintf("%#v", this.TFunc)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringMlp(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *NeuronUnit) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *NeuronUnit) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.Weights) > 0 {
		dAtA[i] = 0xa
		i++
		i = encodeVarintMlp(dAtA, i, uint64(len(m.Weights)*8))
		for _, num := range m.Weights {
			f1 := math.Float64bits(float64(num))
			encoding_binary.LittleEndian.PutUint64(dAtA[i:], uint64(f1))
			i += 8
		}
	}
	if m.Bias != 0 {
		dAtA[i] = 0x11
		i++
		encoding_binary.LittleEndian.PutUint64(dAtA[i:], uint64(math.Float64bits(float64(m.Bias))))
		i += 8
	}
	if m.Lrate != 0 {
		dAtA[i] = 0x19
		i++
		encoding_binary.LittleEndian.PutUint64(dAtA[i:], uint64(math.Float64bits(float64(m.Lrate))))
		i += 8
	}
	if m.Value != 0 {
		dAtA[i] = 0x21
		i++
		encoding_binary.LittleEndian.PutUint64(dAtA[i:], uint64(math.Float64bits(float64(m.Value))))
		i += 8
	}
	if m.Delta != 0 {
		dAtA[i] = 0x29
		i++
		encoding_binary.LittleEndian.PutUint64(dAtA[i:], uint64(math.Float64bits(float64(m.Delta))))
		i += 8
	}
	return i, nil
}

func (m *NeuralLayer) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *NeuralLayer) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.NeuronUnits) > 0 {
		for _, msg := range m.NeuronUnits {
			dAtA[i] = 0xa
			i++
			i = encodeVarintMlp(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	if m.Length != 0 {
		dAtA[i] = 0x10
		i++
		i = encodeVarintMlp(dAtA, i, uint64(m.Length))
	}
	return i, nil
}

func (m *MultiLayerNetwork) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *MultiLayerNetwork) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.LRate != 0 {
		dAtA[i] = 0x9
		i++
		encoding_binary.LittleEndian.PutUint64(dAtA[i:], uint64(math.Float64bits(float64(m.LRate))))
		i += 8
	}
	if len(m.NeuralLayers) > 0 {
		for _, msg := range m.NeuralLayers {
			dAtA[i] = 0x12
			i++
			i = encodeVarintMlp(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	if m.TFunc != 0 {
		dAtA[i] = 0x18
		i++
		i = encodeVarintMlp(dAtA, i, uint64(m.TFunc))
	}
	return i, nil
}

func encodeVarintMlp(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func NewPopulatedNeuronUnit(r randyMlp, easy bool) *NeuronUnit {
	this := &NeuronUnit{}
	v1 := r.Intn(10)
	this.Weights = make([]float64, v1)
	for i := 0; i < v1; i++ {
		this.Weights[i] = float64(r.Float64())
		if r.Intn(2) == 0 {
			this.Weights[i] *= -1
		}
	}
	this.Bias = float64(r.Float64())
	if r.Intn(2) == 0 {
		this.Bias *= -1
	}
	this.Lrate = float64(r.Float64())
	if r.Intn(2) == 0 {
		this.Lrate *= -1
	}
	this.Value = float64(r.Float64())
	if r.Intn(2) == 0 {
		this.Value *= -1
	}
	this.Delta = float64(r.Float64())
	if r.Intn(2) == 0 {
		this.Delta *= -1
	}
	if !easy && r.Intn(10) != 0 {
	}
	return this
}

func NewPopulatedNeuralLayer(r randyMlp, easy bool) *NeuralLayer {
	this := &NeuralLayer{}
	if r.Intn(10) != 0 {
		v2 := r.Intn(5)
		this.NeuronUnits = make([]NeuronUnit, v2)
		for i := 0; i < v2; i++ {
			v3 := NewPopulatedNeuronUnit(r, easy)
			this.NeuronUnits[i] = *v3
		}
	}
	this.Length = int64(r.Int63())
	if r.Intn(2) == 0 {
		this.Length *= -1
	}
	if !easy && r.Intn(10) != 0 {
	}
	return this
}

func NewPopulatedMultiLayerNetwork(r randyMlp, easy bool) *MultiLayerNetwork {
	this := &MultiLayerNetwork{}
	this.LRate = float64(r.Float64())
	if r.Intn(2) == 0 {
		this.LRate *= -1
	}
	if r.Intn(10) != 0 {
		v4 := r.Intn(5)
		this.NeuralLayers = make([]NeuralLayer, v4)
		for i := 0; i < v4; i++ {
			v5 := NewPopulatedNeuralLayer(r, easy)
			this.NeuralLayers[i] = *v5
		}
	}
	this.TFunc = TransferFunc([]int32{0}[r.Intn(1)])
	if !easy && r.Intn(10) != 0 {
	}
	return this
}

type randyMlp interface {
	Float32() float32
	Float64() float64
	Int63() int64
	Int31() int32
	Uint32() uint32
	Intn(n int) int
}

func randUTF8RuneMlp(r randyMlp) rune {
	ru := r.Intn(62)
	if ru < 10 {
		return rune(ru + 48)
	} else if ru < 36 {
		return rune(ru + 55)
	}
	return rune(ru + 61)
}
func randStringMlp(r randyMlp) string {
	v6 := r.Intn(100)
	tmps := make([]rune, v6)
	for i := 0; i < v6; i++ {
		tmps[i] = randUTF8RuneMlp(r)
	}
	return string(tmps)
}
func randUnrecognizedMlp(r randyMlp, maxFieldNumber int) (dAtA []byte) {
	l := r.Intn(5)
	for i := 0; i < l; i++ {
		wire := r.Intn(4)
		if wire == 3 {
			wire = 5
		}
		fieldNumber := maxFieldNumber + r.Intn(100)
		dAtA = randFieldMlp(dAtA, r, fieldNumber, wire)
	}
	return dAtA
}
func randFieldMlp(dAtA []byte, r randyMlp, fieldNumber int, wire int) []byte {
	key := uint32(fieldNumber)<<3 | uint32(wire)
	switch wire {
	case 0:
		dAtA = encodeVarintPopulateMlp(dAtA, uint64(key))
		v7 := r.Int63()
		if r.Intn(2) == 0 {
			v7 *= -1
		}
		dAtA = encodeVarintPopulateMlp(dAtA, uint64(v7))
	case 1:
		dAtA = encodeVarintPopulateMlp(dAtA, uint64(key))
		dAtA = append(dAtA, byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)))
	case 2:
		dAtA = encodeVarintPopulateMlp(dAtA, uint64(key))
		ll := r.Intn(100)
		dAtA = encodeVarintPopulateMlp(dAtA, uint64(ll))
		for j := 0; j < ll; j++ {
			dAtA = append(dAtA, byte(r.Intn(256)))
		}
	default:
		dAtA = encodeVarintPopulateMlp(dAtA, uint64(key))
		dAtA = append(dAtA, byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)))
	}
	return dAtA
}
func encodeVarintPopulateMlp(dAtA []byte, v uint64) []byte {
	for v >= 1<<7 {
		dAtA = append(dAtA, uint8(uint64(v)&0x7f|0x80))
		v >>= 7
	}
	dAtA = append(dAtA, uint8(v))
	return dAtA
}
func (m *NeuronUnit) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Weights) > 0 {
		n += 1 + sovMlp(uint64(len(m.Weights)*8)) + len(m.Weights)*8
	}
	if m.Bias != 0 {
		n += 9
	}
	if m.Lrate != 0 {
		n += 9
	}
	if m.Value != 0 {
		n += 9
	}
	if m.Delta != 0 {
		n += 9
	}
	return n
}

func (m *NeuralLayer) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.NeuronUnits) > 0 {
		for _, e := range m.NeuronUnits {
			l = e.Size()
			n += 1 + l + sovMlp(uint64(l))
		}
	}
	if m.Length != 0 {
		n += 1 + sovMlp(uint64(m.Length))
	}
	return n
}

func (m *MultiLayerNetwork) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.LRate != 0 {
		n += 9
	}
	if len(m.NeuralLayers) > 0 {
		for _, e := range m.NeuralLayers {
			l = e.Size()
			n += 1 + l + sovMlp(uint64(l))
		}
	}
	if m.TFunc != 0 {
		n += 1 + sovMlp(uint64(m.TFunc))
	}
	return n
}

func sovMlp(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozMlp(x uint64) (n int) {
	return sovMlp(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *NeuronUnit) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&NeuronUnit{`,
		`Weights:` + fmt.Sprintf("%v", this.Weights) + `,`,
		`Bias:` + fmt.Sprintf("%v", this.Bias) + `,`,
		`Lrate:` + fmt.Sprintf("%v", this.Lrate) + `,`,
		`Value:` + fmt.Sprintf("%v", this.Value) + `,`,
		`Delta:` + fmt.Sprintf("%v", this.Delta) + `,`,
		`}`,
	}, "")
	return s
}
func (this *NeuralLayer) String() string {
	if this == nil {
		return "nil"
	}
	repeatedStringForNeuronUnits := "[]NeuronUnit{"
	for _, f := range this.NeuronUnits {
		repeatedStringForNeuronUnits += strings.Replace(strings.Replace(f.String(), "NeuronUnit", "NeuronUnit", 1), `&`, ``, 1) + ","
	}
	repeatedStringForNeuronUnits += "}"
	s := strings.Join([]string{`&NeuralLayer{`,
		`NeuronUnits:` + repeatedStringForNeuronUnits + `,`,
		`Length:` + fmt.Sprintf("%v", this.Length) + `,`,
		`}`,
	}, "")
	return s
}
func (this *MultiLayerNetwork) String() string {
	if this == nil {
		return "nil"
	}
	repeatedStringForNeuralLayers := "[]NeuralLayer{"
	for _, f := range this.NeuralLayers {
		repeatedStringForNeuralLayers += strings.Replace(strings.Replace(f.String(), "NeuralLayer", "NeuralLayer", 1), `&`, ``, 1) + ","
	}
	repeatedStringForNeuralLayers += "}"
	s := strings.Join([]string{`&MultiLayerNetwork{`,
		`LRate:` + fmt.Sprintf("%v", this.LRate) + `,`,
		`NeuralLayers:` + repeatedStringForNeuralLayers + `,`,
		`TFunc:` + fmt.Sprintf("%v", this.TFunc) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringMlp(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *NeuronUnit) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowMlp
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: NeuronUnit: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: NeuronUnit: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType == 1 {
				var v uint64
				if (iNdEx + 8) > l {
					return io.ErrUnexpectedEOF
				}
				v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
				iNdEx += 8
				v2 := float64(math.Float64frombits(v))
				m.Weights = append(m.Weights, v2)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowMlp
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					packedLen |= int(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				if packedLen < 0 {
					return ErrInvalidLengthMlp
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return ErrInvalidLengthMlp
				}
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				var elementCount int
				elementCount = packedLen / 8
				if elementCount != 0 && len(m.Weights) == 0 {
					m.Weights = make([]float64, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v uint64
					if (iNdEx + 8) > l {
						return io.ErrUnexpectedEOF
					}
					v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
					iNdEx += 8
					v2 := float64(math.Float64frombits(v))
					m.Weights = append(m.Weights, v2)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field Weights", wireType)
			}
		case 2:
			if wireType != 1 {
				return fmt.Errorf("proto: wrong wireType = %d for field Bias", wireType)
			}
			var v uint64
			if (iNdEx + 8) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
			iNdEx += 8
			m.Bias = float64(math.Float64frombits(v))
		case 3:
			if wireType != 1 {
				return fmt.Errorf("proto: wrong wireType = %d for field Lrate", wireType)
			}
			var v uint64
			if (iNdEx + 8) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
			iNdEx += 8
			m.Lrate = float64(math.Float64frombits(v))
		case 4:
			if wireType != 1 {
				return fmt.Errorf("proto: wrong wireType = %d for field Value", wireType)
			}
			var v uint64
			if (iNdEx + 8) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
			iNdEx += 8
			m.Value = float64(math.Float64frombits(v))
		case 5:
			if wireType != 1 {
				return fmt.Errorf("proto: wrong wireType = %d for field Delta", wireType)
			}
			var v uint64
			if (iNdEx + 8) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
			iNdEx += 8
			m.Delta = float64(math.Float64frombits(v))
		default:
			iNdEx = preIndex
			skippy, err := skipMlp(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthMlp
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthMlp
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func (m *NeuralLayer) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowMlp
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: NeuralLayer: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: NeuralLayer: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field NeuronUnits", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMlp
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthMlp
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthMlp
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.NeuronUnits = append(m.NeuronUnits, NeuronUnit{})
			if err := m.NeuronUnits[len(m.NeuronUnits)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Length", wireType)
			}
			m.Length = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMlp
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Length |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipMlp(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthMlp
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthMlp
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func (m *MultiLayerNetwork) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowMlp
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: MultiLayerNetwork: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: MultiLayerNetwork: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 1 {
				return fmt.Errorf("proto: wrong wireType = %d for field LRate", wireType)
			}
			var v uint64
			if (iNdEx + 8) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
			iNdEx += 8
			m.LRate = float64(math.Float64frombits(v))
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field NeuralLayers", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMlp
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthMlp
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthMlp
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.NeuralLayers = append(m.NeuralLayers, NeuralLayer{})
			if err := m.NeuralLayers[len(m.NeuralLayers)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field TFunc", wireType)
			}
			m.TFunc = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMlp
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.TFunc |= TransferFunc(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipMlp(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthMlp
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthMlp
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func skipMlp(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowMlp
			}
			if iNdEx >= l {
				return 0, io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		wireType := int(wire & 0x7)
		switch wireType {
		case 0:
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowMlp
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				iNdEx++
				if dAtA[iNdEx-1] < 0x80 {
					break
				}
			}
			return iNdEx, nil
		case 1:
			iNdEx += 8
			return iNdEx, nil
		case 2:
			var length int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowMlp
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				length |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if length < 0 {
				return 0, ErrInvalidLengthMlp
			}
			iNdEx += length
			if iNdEx < 0 {
				return 0, ErrInvalidLengthMlp
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowMlp
					}
					if iNdEx >= l {
						return 0, io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					innerWire |= (uint64(b) & 0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				innerWireType := int(innerWire & 0x7)
				if innerWireType == 4 {
					break
				}
				next, err := skipMlp(dAtA[start:])
				if err != nil {
					return 0, err
				}
				iNdEx = start + next
				if iNdEx < 0 {
					return 0, ErrInvalidLengthMlp
				}
			}
			return iNdEx, nil
		case 4:
			return iNdEx, nil
		case 5:
			iNdEx += 4
			return iNdEx, nil
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
	}
	panic("unreachable")
}

var (
	ErrInvalidLengthMlp = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowMlp   = fmt.Errorf("proto: integer overflow")
)
