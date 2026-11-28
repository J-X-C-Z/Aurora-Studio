//! Zero-dependency ONNX Protobuf wire-format decoder.
//!
//! Decodes `ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`,
//! `AttributeProto`, `ValueInfoProto`, and `OperatorSetIdProto` directly from
//! `.onnx` bytes without `prost` or `protobuf` crates.
//!
//! Supports `raw_data`, `float_data`, `int64_data`, `int32_data`, `double_data`,
//! as well as FP32 / FP16 / INT64 / INT32 / UINT8 / BOOL / DOUBLE tensors and
//! nested `GraphProto` attributes (e.g. `If` `then_branch` / `else_branch`).

use std::fmt;

/// ONNX `TensorProto.DataType` enum values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DataType {
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    BFloat16 = 16,
}

impl DataType {
    pub fn from_u32(v: u32) -> Result<Self, ProtoError> {
        match v {
            0 => Ok(Self::Undefined),
            1 => Ok(Self::Float),
            2 => Ok(Self::Uint8),
            3 => Ok(Self::Int8),
            4 => Ok(Self::Uint16),
            5 => Ok(Self::Int16),
            6 => Ok(Self::Int32),
            7 => Ok(Self::Int64),
            8 => Ok(Self::String),
            9 => Ok(Self::Bool),
            10 => Ok(Self::Float16),
            11 => Ok(Self::Double),
            12 => Ok(Self::Uint32),
            13 => Ok(Self::Uint64),
            16 => Ok(Self::BFloat16),
            other => Err(ProtoError::UnsupportedDataType(other)),
        }
    }

    /// Size of one element in bytes, or `None` for undefined/variable-length types.
    pub const fn byte_size(self) -> Option<usize> {
        match self {
            Self::Float | Self::Int32 | Self::Uint32 => Some(4),
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 | Self::Float16 | Self::BFloat16 => Some(2),
            Self::Int64 | Self::Uint64 | Self::Double => Some(8),
            Self::Undefined | Self::String => None,
        }
    }
}

/// Protobuf wire decoding error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProtoError {
    UnexpectedEof,
    VarintOverflow,
    InvalidWireType(u8),
    InvalidUtf8,
    UnsupportedDataType(u32),
    ExternalDataUnsupported(String),
    MissingGraph,
    InvalidDataLength { expected: usize, actual: usize },
}

impl fmt::Display for ProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected EOF while decoding ONNX protobuf"),
            Self::VarintOverflow => write!(f, "varint overflow while decoding ONNX protobuf"),
            Self::InvalidWireType(w) => write!(f, "invalid protobuf wire type: {w}"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 string in ONNX protobuf"),
            Self::UnsupportedDataType(dt) => write!(f, "unsupported ONNX TensorProto data_type: {dt}"),
            Self::ExternalDataUnsupported(name) => {
                write!(f, "external tensor data is not supported: {name}")
            }
            Self::MissingGraph => write!(f, "ONNX ModelProto has no GraphProto"),
            Self::InvalidDataLength { expected, actual } => {
                write!(f, "invalid tensor byte length: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for ProtoError {}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelProto {
    pub ir_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub opset_imports: Vec<OperatorSetIdProto>,
    pub graph: GraphProto,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorSetIdProto {
    pub domain: String,
    pub version: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphProto {
    pub name: String,
    pub nodes: Vec<NodeProto>,
    pub initializers: Vec<TensorProto>,
    pub inputs: Vec<ValueInfoProto>,
    pub outputs: Vec<ValueInfoProto>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeProto {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub attributes: Vec<AttributeProto>,
}

impl NodeProto {
    pub fn attr(&self, name: &str) -> Option<&AttributeProto> {
        self.attributes.iter().find(|a| a.name == name)
    }

    pub fn attr_int(&self, name: &str) -> Option<i64> {
        match &self.attr(name)?.value {
            AttributeValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn attr_float(&self, name: &str) -> Option<f32> {
        match &self.attr(name)?.value {
            AttributeValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn attr_string(&self, name: &str) -> Option<&str> {
        match &self.attr(name)?.value {
            AttributeValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn attr_ints(&self, name: &str) -> Option<&[i64]> {
        match &self.attr(name)?.value {
            AttributeValue::Ints(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn attr_tensor(&self, name: &str) -> Option<&TensorProto> {
        match &self.attr(name)?.value {
            AttributeValue::Tensor(t) => Some(t),
            _ => None,
        }
    }

    pub fn attr_graph(&self, name: &str) -> Option<&GraphProto> {
        match &self.attr(name)?.value {
            AttributeValue::Graph(g) => Some(g),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttributeProto {
    pub name: String,
    pub value: AttributeValue,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AttributeValue {
    Undefined,
    Float(f32),
    Int(i64),
    String(String),
    Tensor(TensorProto),
    Graph(GraphProto),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DimSpec {
    Value(i64),
    Param(String),
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueInfoProto {
    pub name: String,
    pub data_type: DataType,
    pub dims: Vec<DimSpec>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorProto {
    pub name: String,
    pub dims: Vec<i64>,
    pub data_type: DataType,
    /// Little-endian raw tensor payload normalized from `raw_data` or typed
    /// repeated fields (`float_data`, `int64_data`, `int32_data`, `double_data`).
    pub raw_data: Vec<u8>,
}

impl TensorProto {
    /// Total element count implied by `dims` (scalar `[]` has 1 element).
    pub fn elem_count(&self) -> usize {
        self.dims.iter().fold(1usize, |acc, &d| {
            acc.saturating_mul(if d > 0 { d as usize } else { 0 })
        })
    }

    /// Decode payload into `Vec<f32>` (supports `Float`, `Float16`, `Double`, `Int32`, `Int64`, `Uint8`).
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, ProtoError> {
        match self.data_type {
            DataType::Float => {
                if self.raw_data.len() % 4 != 0 {
                    return Err(ProtoError::InvalidDataLength {
                        expected: self.elem_count() * 4,
                        actual: self.raw_data.len(),
                    });
                }
                Ok(self
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            DataType::Float16 => {
                if self.raw_data.len() % 2 != 0 {
                    return Err(ProtoError::InvalidDataLength {
                        expected: self.elem_count() * 2,
                        actual: self.raw_data.len(),
                    });
                }
                Ok(self
                    .raw_data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect())
            }
            DataType::Double => {
                if self.raw_data.len() % 8 != 0 {
                    return Err(ProtoError::InvalidDataLength {
                        expected: self.elem_count() * 8,
                        actual: self.raw_data.len(),
                    });
                }
                Ok(self
                    .raw_data
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect())
            }
            DataType::Int32 => Ok(self
                .raw_data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect()),
            DataType::Int64 => Ok(self
                .raw_data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect()),
            DataType::Uint8 => Ok(self.raw_data.iter().map(|&b| b as f32).collect()),
            other => Err(ProtoError::UnsupportedDataType(other as u32)),
        }
    }

    /// Decode payload into `Vec<i64>` (supports `Int64` and `Int32`).
    pub fn to_i64_vec(&self) -> Result<Vec<i64>, ProtoError> {
        match self.data_type {
            DataType::Int64 => {
                if self.raw_data.len() % 8 != 0 {
                    return Err(ProtoError::InvalidDataLength {
                        expected: self.elem_count() * 8,
                        actual: self.raw_data.len(),
                    });
                }
                Ok(self
                    .raw_data
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect())
            }
            DataType::Int32 => {
                if self.raw_data.len() % 4 != 0 {
                    return Err(ProtoError::InvalidDataLength {
                        expected: self.elem_count() * 4,
                        actual: self.raw_data.len(),
                    });
                }
                Ok(self
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                    .collect())
            }
            other => Err(ProtoError::UnsupportedDataType(other as u32)),
        }
    }

    /// Construct a `TensorProto` from `f32` values.
    pub fn from_f32(name: impl Into<String>, dims: Vec<i64>, values: &[f32]) -> Self {
        let mut raw_data = Vec::with_capacity(values.len() * 4);
        for &v in values {
            raw_data.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            name: name.into(),
            dims,
            data_type: DataType::Float,
            raw_data,
        }
    }

    /// Construct a `TensorProto` from `i64` values.
    pub fn from_i64(name: impl Into<String>, dims: Vec<i64>, values: &[i64]) -> Self {
        let mut raw_data = Vec::with_capacity(values.len() * 8);
        for &v in values {
            raw_data.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            name: name.into(),
            dims,
            data_type: DataType::Int64,
            raw_data,
        }
    }
}

/// IEEE-754 binary16 (FP16) to binary32 (FP32) conversion without external crates.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits as u32) >> 10) & 0x1f;
    let frac = (bits as u32) & 0x03ff;

    let out_bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // Subnormal half -> normalized f32
            let mut m = frac;
            let mut e: i32 = -14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            let f32_exp = ((e + 127) as u32) << 23;
            sign | f32_exp | (m << 13)
        }
    } else if exp == 0x1f {
        // Inf / NaN
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        let f32_exp = (exp + 112) << 23;
        sign | f32_exp | (frac << 13)
    };
    f32::from_bits(out_bits)
}

/// IEEE-754 binary32 (FP32) to binary16 (FP16) round-to-nearest-even conversion.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    if exponent == 0xFF {
        let payload = if mantissa != 0 {
            0x200 | ((mantissa >> 13) as u16 & 0x3FF)
        } else {
            0
        };
        return sign | 0x7C00 | payload;
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1F {
        return sign | 0x7C00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = (14 - half_exponent) as u32;
        let mut half = (mantissa >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        if mantissa & round_bit != 0
            && ((mantissa & (round_bit - 1)) != 0 || half & 1 != 0)
        {
            half += 1;
        }
        return sign | half;
    }
    let mut half = (((half_exponent as u32) << 10) as u16) | ((mantissa >> 13) as u16);
    let round = mantissa & 0x1FFF;
    if round > 0x1000 || (round == 0x1000 && half & 1 != 0) {
        half += 1;
    }
    sign | half
}

#[cfg(test)]
#[test]
fn fp16_midpoints_round_to_even_on_both_sides_of_every_finite_interval(){
    for low in 0u16..0x7bff {
        let hi=low+1;
        // Every binary16 value and its exact midpoint fit strictly within f32 precision.
        let left=f16_to_f32(low);let right=f16_to_f32(hi);let m=(left+right)*0.5;
        for (sign,mask) in [(1.0f32,0u16),(-1.0,0x8000)] {
            for (x,want) in [(left,low),(f32::from_bits(m.to_bits()-1),low),
                (m,if low&1==0 {low}else{hi}),(f32::from_bits(m.to_bits()+1),hi),(right,hi)] {
                assert_eq!(f32_to_f16(sign*x),want|mask,"interval {low:04x}, x={x}, sign={sign}");
            }
        }
    }
    assert_eq!(f32_to_f16(65520.0),0x7c00);assert_eq!(f32_to_f16(-65520.0),0xfc00);
    assert_eq!(f32_to_f16(f32::from_bits(0x477f_efff)),0x7bff);
    assert_eq!(f32_to_f16(-0.),0x8000);
    assert_ne!(f32_to_f16(f32::NAN)&0x03ff,0);
}

// ---------------------------------------------------------------------------
// Low-level protobuf wire reader
// ---------------------------------------------------------------------------

struct WireReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> WireReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn read_varint(&mut self) -> Result<u64, ProtoError> {
        let mut result = 0u64;
        let mut shift = 0u32;
        while self.pos < self.buf.len() {
            let b = self.buf[self.pos];
            self.pos += 1;
            if shift >= 64 {
                return Err(ProtoError::VarintOverflow);
            }
            result |= ((b & 0x7f) as u64) << shift;
            if (b & 0x80) == 0 {
                return Ok(result);
            }
            shift += 7;
        }
        Err(ProtoError::UnexpectedEof)
    }

    fn read_tag(&mut self) -> Result<(u32, u8), ProtoError> {
        let tag = self.read_varint()?;
        Ok(((tag >> 3) as u32, (tag & 0x7) as u8))
    }

    fn read_bytes(&mut self) -> Result<&'a [u8], ProtoError> {
        let len = self.read_varint()? as usize;
        let end = self.pos.checked_add(len).ok_or(ProtoError::UnexpectedEof)?;
        if end > self.buf.len() {
            return Err(ProtoError::UnexpectedEof);
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_string(&mut self) -> Result<String, ProtoError> {
        let bytes = self.read_bytes()?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_owned())
            .map_err(|_| ProtoError::InvalidUtf8)
    }

    fn read_fixed32(&mut self) -> Result<[u8; 4], ProtoError> {
        let end = self.pos.checked_add(4).ok_or(ProtoError::UnexpectedEof)?;
        if end > self.buf.len() {
            return Err(ProtoError::UnexpectedEof);
        }
        let mut out = [0u8; 4];
        out.copy_from_slice(&self.buf[self.pos..end]);
        self.pos = end;
        Ok(out)
    }

    fn read_fixed64(&mut self) -> Result<[u8; 8], ProtoError> {
        let end = self.pos.checked_add(8).ok_or(ProtoError::UnexpectedEof)?;
        if end > self.buf.len() {
            return Err(ProtoError::UnexpectedEof);
        }
        let mut out = [0u8; 8];
        out.copy_from_slice(&self.buf[self.pos..end]);
        self.pos = end;
        Ok(out)
    }

    fn skip_field(&mut self, wire_type: u8) -> Result<(), ProtoError> {
        match wire_type {
            0 => {
                let _ = self.read_varint()?;
                Ok(())
            }
            1 => {
                let _ = self.read_fixed64()?;
                Ok(())
            }
            2 => {
                let _ = self.read_bytes()?;
                Ok(())
            }
            5 => {
                let _ = self.read_fixed32()?;
                Ok(())
            }
            other => Err(ProtoError::InvalidWireType(other)),
        }
    }
}

fn read_packed_varints_i64(bytes: &[u8], out: &mut Vec<i64>) -> Result<(), ProtoError> {
    let mut r = WireReader::new(bytes);
    while !r.is_eof() {
        out.push(r.read_varint()? as i64);
    }
    Ok(())
}

fn read_packed_f32(bytes: &[u8], out: &mut Vec<f32>) -> Result<(), ProtoError> {
    if bytes.len() % 4 != 0 {
        return Err(ProtoError::UnexpectedEof);
    }
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Decode an ONNX `ModelProto` from raw `.onnx` bytes.
pub fn decode_model(bytes: &[u8]) -> Result<ModelProto, ProtoError> {
    let mut r = WireReader::new(bytes);
    let mut ir_version = 0i64;
    let mut producer_name = String::new();
    let mut producer_version = String::new();
    let mut opset_imports = Vec::new();
    let mut graph = None;

    while !r.is_eof() {
        let (field, wt) = r.read_tag()?;
        match (field, wt) {
            (1, 0) => ir_version = r.read_varint()? as i64,
            (2, 2) => producer_name = r.read_string()?,
            (3, 2) => producer_version = r.read_string()?,
            (7, 2) => graph = Some(decode_graph(r.read_bytes()?)?),
            (8, 2) => opset_imports.push(decode_opset_id(r.read_bytes()?)?),
            (_, w) => r.skip_field(w)?,
        }
    }

    Ok(ModelProto {
        ir_version,
        producer_name,
        producer_version,
        opset_imports,
        graph: graph.ok_or(ProtoError::MissingGraph)?,
    })
}

fn decode_opset_id(bytes: &[u8]) -> Result<OperatorSetIdProto, ProtoError> {
    let mut r = WireReader::new(bytes);
    let mut domain = String::new();
    let mut version = 0i64;
    while !r.is_eof() {
        let (field, wt) = r.read_tag()?;
        match (field, wt) {
            (1, 2) => domain = r.read_string()?,
            (2, 0) => version = r.read_varint()? as i64,
            (_, w) => r.skip_field(w)?,
        }
    }
    Ok(OperatorSetIdProto { domain, version })
}

fn decode_graph(bytes: &[u8]) -> Result<GraphProto, ProtoError> {
    let mut r = WireReader::new(bytes);
    let mut name = String::new();
    let mut nodes = Vec::new();
    let mut initializers = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    while !r.is_eof() {
        let (field, wt) = r.read_tag()?;
        match (field, wt) {
            (1, 2) => nodes.push(decode_node(r.read_bytes()?)?),
            (2, 2) => name = r.read_string()?,
            (5, 2) => initializers.push(decode_tensor(r.read_bytes()?)?),
            (11, 2) => inputs.push(decode_value_info(r.read_bytes()?)?),
            (12, 2) => outputs.push(decode_value_info(r.read_bytes()?)?),
            (_, w) => r.skip_field(w)?,
        }
    }

    Ok(GraphProto {
        name,
        nodes,
        initializers,
        inputs,
        outputs,
    })
}

fn decode_node(bytes: &[u8]) -> Result<NodeProto, ProtoError> {
    let mut r = WireReader::new(bytes);
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut name = String::new();
    let mut op_type = String::new();
    let mut domain = String::new();
    let mut attributes = Vec::new();

    while !r.is_eof() {
        let (field, wt) = r.read_tag()?;
        match (field, wt) {
            (1, 2) => inputs.push(r.read_string()?),
            (2, 2) => outputs.push(r.read_string()?),
            (3, 2) => name = r.read_string()?,
            (4, 2) => op_type = r.read_string()?,
            (5, 2) => attributes.push(decode_attribute(r.read_bytes()?)?),
            (7, 2) => domain = r.read_string()?,
            (_, w) => r.skip_field(w)?,
        }
    }

    Ok(NodeProto {
        inputs,
        outputs,
        name,
        op_type,
        domain,
        attributes,
    })
}

fn decode_attribute(bytes: &[u8]) -> Result<AttributeProto, ProtoError> {
    let mut r = WireReader::new(bytes);
    let mut name = String::new();
    let mut atype = 0u32;
    let mut f_val: Option<f32> = None;
    let mut i_val: Option<i64> = None;
    let mut s_val: Option<String> = None;
    let mut t_val: Option<TensorProto> = None;
    let mut g_val: Option<GraphProto> = None;
    let mut floats = Vec::new();
    let mut ints = Vec::new();
    let mut strings = Vec::new();

    while !r.is_eof() {
        let (field, wt) = r.read_tag()?;
        match (field, wt) {
            (1, 2) => name = r.read_string()?,
            (2, 5) => f_val = Some(f32::from_le_bytes(r.read_fixed32()?)),
            (3, 0) => i_val = Some(r.read_varint()? as i64),
            (4, 2) => s_val = Some(r.read_string()?),
            (5, 2) => t_val = Some(decode_tensor(r.read_bytes()?)?),
            (6, 2) => g_val = Some(decode_graph(r.read_bytes()?)?),
            (7, 5) => floats.push(f32::from_le_bytes(r.read_fixed32()?)),
            (7, 2) => read_packed_f32(r.read_bytes()?, &mut floats)?,
            (8, 0) => ints.push(r.read_varint()? as i64),
            (8, 2) => read_packed_varints_i64(r.read_bytes()?, &mut ints)?,
            (9, 2) => strings.push(r.read_string()?),
            (20, 0) => atype = r.read_varint()? as u32,
            (_, w) => r.skip_field(w)?,
        }
    }

    let value = match atype {
        1 => AttributeValue::Float(f_val.unwrap_or(0.0)),
        2 => AttributeValue::Int(i_val.unwrap_or(0)),
        3 => AttributeValue::String(s_val.unwrap_or_default()),
        4 => match t_val {
            Some(t) => AttributeValue::Tensor(t),
            None => AttributeValue::Undefined,
        },
        5 => match g_val {
            Some(g) => AttributeValue::Graph(g),
            None => AttributeValue::Undefined,
        },
        6 => AttributeValue::Floats(floats),
        7 => AttributeValue::Ints(ints),
        8 => AttributeValue::Strings(strings),
        _ => {
            // Fallback when explicit `type` tag is omitted.
            if let Some(g) = g_val {
                AttributeValue::Graph(g)
            } else if let Some(t) = t_val {
                AttributeValue::Tensor(t)
            } else if !ints.is_empty() {
                AttributeValue::Ints(ints)
            } else if !floats.is_empty() {
                AttributeValue::Floats(floats)
            } else if !strings.is_empty() {
                AttributeValue::Strings(strings)
            } else if let Some(s) = s_val {
                AttributeValue::String(s)
            } else if let Some(i) = i_val {
                AttributeValue::Int(i)
            } else if let Some(f) = f_val {
                AttributeValue::Float(f)
            } else {
                AttributeValue::Undefined
            }
        }
    };

    Ok(AttributeProto { name, value })
}

fn decode_tensor(bytes: &[u8]) -> Result<TensorProto, ProtoError> {
    let mut r = WireReader::new(bytes);
    let mut dims = Vec::new();
    let mut data_type = DataType::Undefined;
    let mut name = String::new();
    let mut raw_data: Option<Vec<u8>> = None;
    let mut float_data: Vec<f32> = Vec::new();
    let mut int32_data: Vec<i32> = Vec::new();
    let mut int64_data: Vec<i64> = Vec::new();
    let mut double_data: Vec<f64> = Vec::new();
    let mut uint64_data: Vec<u64> = Vec::new();
    let mut data_location = 0u32;

    while !r.is_eof() {
        let (field, wt) = r.read_tag()?;
        match (field, wt) {
            (1, 0) => dims.push(r.read_varint()? as i64),
            (1, 2) => read_packed_varints_i64(r.read_bytes()?, &mut dims)?,
            (2, 0) => data_type = DataType::from_u32(r.read_varint()? as u32)?,
            (4, 5) => float_data.push(f32::from_le_bytes(r.read_fixed32()?)),
            (4, 2) => read_packed_f32(r.read_bytes()?, &mut float_data)?,
            (5, 0) => int32_data.push(r.read_varint()? as i32),
            (5, 2) => {
                let mut tmp = Vec::new();
                read_packed_varints_i64(r.read_bytes()?, &mut tmp)?;
                int32_data.extend(tmp.into_iter().map(|v| v as i32));
            }
            (7, 0) => int64_data.push(r.read_varint()? as i64),
            (7, 2) => read_packed_varints_i64(r.read_bytes()?, &mut int64_data)?,
            (8, 2) => name = r.read_string()?,
            (9, 2) => raw_data = Some(r.read_bytes()?.to_vec()),
            (10, 1) => double_data.push(f64::from_le_bytes(r.read_fixed64()?)),
            (10, 2) => {
                let b = r.read_bytes()?;
                if b.len() % 8 != 0 {
                    return Err(ProtoError::UnexpectedEof);
                }
                for c in b.chunks_exact(8) {
                    double_data.push(f64::from_le_bytes(c.try_into().unwrap()));
                }
            }
            (11, 0) => uint64_data.push(r.read_varint()?),
            (11, 2) => {
                let mut pr = WireReader::new(r.read_bytes()?);
                while !pr.is_eof() {
                    uint64_data.push(pr.read_varint()?);
                }
            }
            (14, 0) => data_location = r.read_varint()? as u32,
            (_, w) => r.skip_field(w)?,
        }
    }

    if data_location == 1 {
        return Err(ProtoError::ExternalDataUnsupported(name));
    }

    // Normalize typed repeated fields into little-endian `raw_data`.
    let raw_data = if let Some(raw) = raw_data {
        raw
    } else if !float_data.is_empty() {
        let mut out = Vec::with_capacity(float_data.len() * 4);
        for v in float_data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    } else if !int64_data.is_empty() {
        let mut out = Vec::with_capacity(int64_data.len() * 8);
        for v in int64_data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    } else if !int32_data.is_empty() {
        // ONNX stores FLOAT16/UINT8/INT8/INT16/BOOL in `int32_data` when not using `raw_data`.
        match data_type {
            DataType::Float16 | DataType::Uint16 | DataType::Int16 | DataType::BFloat16 => {
                let mut out = Vec::with_capacity(int32_data.len() * 2);
                for v in int32_data {
                    out.extend_from_slice(&(v as u16).to_le_bytes());
                }
                out
            }
            DataType::Uint8 | DataType::Int8 | DataType::Bool => {
                int32_data.into_iter().map(|v| v as u8).collect()
            }
            _ => {
                let mut out = Vec::with_capacity(int32_data.len() * 4);
                for v in int32_data {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
        }
    } else if !double_data.is_empty() {
        let mut out = Vec::with_capacity(double_data.len() * 8);
        for v in double_data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    } else if !uint64_data.is_empty() {
        let mut out = Vec::with_capacity(uint64_data.len() * 8);
        for v in uint64_data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    } else {
        Vec::new()
    };

    Ok(TensorProto {
        name,
        dims,
        data_type,
        raw_data,
    })
}

fn decode_value_info(bytes: &[u8]) -> Result<ValueInfoProto, ProtoError> {
    let mut r = WireReader::new(bytes);
    let mut name = String::new();
    let mut data_type = DataType::Undefined;
    let mut dims = Vec::new();

    while !r.is_eof() {
        let (field, wt) = r.read_tag()?;
        match (field, wt) {
            (1, 2) => name = r.read_string()?,
            (2, 2) => {
                let (dt, ds) = decode_type_proto(r.read_bytes()?)?;
                data_type = dt;
                dims = ds;
            }
            (_, w) => r.skip_field(w)?,
        }
    }

    Ok(ValueInfoProto {
        name,
        data_type,
        dims,
    })
}

fn decode_type_proto(bytes: &[u8]) -> Result<(DataType, Vec<DimSpec>), ProtoError> {
    let mut r = WireReader::new(bytes);
    let mut data_type = DataType::Undefined;
    let mut dims = Vec::new();

    while !r.is_eof() {
        let (field, wt) = r.read_tag()?;
        match (field, wt) {
            (1, 2) => {
                let mut tr = WireReader::new(r.read_bytes()?);
                while !tr.is_eof() {
                    let (tf, tw) = tr.read_tag()?;
                    match (tf, tw) {
                        (1, 0) => data_type = DataType::from_u32(tr.read_varint()? as u32)?,
                        (2, 2) => dims = decode_tensor_shape(tr.read_bytes()?)?,
                        (_, w) => tr.skip_field(w)?,
                    }
                }
            }
            (_, w) => r.skip_field(w)?,
        }
    }

    Ok((data_type, dims))
}

fn decode_tensor_shape(bytes: &[u8]) -> Result<Vec<DimSpec>, ProtoError> {
    let mut r = WireReader::new(bytes);
    let mut dims = Vec::new();
    while !r.is_eof() {
        let (field, wt) = r.read_tag()?;
        match (field, wt) {
            (1, 2) => {
                let mut dr = WireReader::new(r.read_bytes()?);
                let mut spec = DimSpec::Unknown;
                while !dr.is_eof() {
                    let (df, dw) = dr.read_tag()?;
                    match (df, dw) {
                        (1, 0) => spec = DimSpec::Value(dr.read_varint()? as i64),
                        (2, 2) => spec = DimSpec::Param(dr.read_string()?),
                        (_, w) => dr.skip_field(w)?,
                    }
                }
                dims.push(spec);
            }
            (_, w) => r.skip_field(w)?,
        }
    }
    Ok(dims)
}
