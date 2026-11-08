use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::Debug;
use std::io::{self, Write};
use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBytes, PyDict, PyString};
use pyo3::{IntoPyObjectExt, prelude::*};
use pyo3::{PyTraverseError, PyVisit, intern};

use enum_dispatch::enum_dispatch;
use serde::ser::SerializeMap as _;
use serde::{Serialize, Serializer};
use serde_json::ser::{Formatter, PrettyFormatter};

use crate::build_tools::py_schema_err;
use crate::build_tools::py_schema_error_type;
use crate::definitions::DefinitionsBuilder;
use crate::py_gc::PyGcTraverse;
use crate::serializers::SerMode;
use crate::serializers::errors::WrappedSerError;
use crate::serializers::polymorphism_trampoline::PolymorphismTrampoline;
use crate::serializers::ser::PythonSerializer;
use crate::serializers::type_serializers::any::AnySerializer;
use crate::tools::{SchemaDict, py_err};

use super::errors::se_err_py_err;
use super::extra::SerializationState;
use super::infer::{infer_json_key, infer_serialize, infer_to_python};

pub(crate) trait BuildSerializer: Sized {
    const EXPECTED_TYPE: &'static str;

    fn build(
        schema: &Bound<'_, PyDict>,
        config: Option<&Bound<'_, PyDict>>,
        definitions: &mut DefinitionsBuilder<Arc<CombinedSerializer>>,
    ) -> PyResult<Arc<CombinedSerializer>>;
}

/// Build the `CombinedSerializer` enum and implement a `find_serializer` method for it.
macro_rules! combined_serializer {
    (
        enum_only: {$($e_key:ident: $e_serializer:path;)*}
        find_only: {$($builder:path;)*}
        both: {$($b_key:ident: $b_serializer:path;)*}
    ) => {
        #[derive(Debug)]
        #[enum_dispatch]
        pub enum CombinedSerializer {
            $($e_key($e_serializer),)*
            $($b_key($b_serializer),)*
        }

        impl CombinedSerializer {
            fn find_serializer(
                lookup_type: &str,
                schema: &Bound<'_, PyDict>,
                config: Option<&Bound<'_, PyDict>>,
                definitions: &mut DefinitionsBuilder<Arc<CombinedSerializer>>
            ) -> PyResult<Arc<CombinedSerializer>> {
                match lookup_type {
                    $(
                        <$b_serializer>::EXPECTED_TYPE => match <$b_serializer>::build(schema, config, definitions) {
                            Ok(serializer) => Ok(serializer),
                            Err(err) => py_schema_err!("Error building `{lookup_type}` serializer:\n  {err}"),
                        },
                    )*
                    $(
                        <$builder>::EXPECTED_TYPE => match <$builder>::build(schema, config, definitions) {
                            Ok(serializer) => Ok(serializer),
                            Err(err) => py_schema_err!("Error building `{lookup_type}` serializer:\n  {err}"),
                        },
                    )*
                    _ => py_schema_err!("Unknown serialization schema type: `{lookup_type}`"),
                }
            }
        }

    };
}

combined_serializer! {
    // `enum_only` is for type_serializers which are not built directly via the `type` key and `find_serializer`
    // but are included in the `CombinedSerializer` enum
    enum_only: {
        // function type_serializers cannot be defined by type lookup, but must be members of `CombinedSerializer`,
        // hence they're here.
        Function: super::type_serializers::function::FunctionPlainSerializer;
        FunctionWrap: super::type_serializers::function::FunctionWrapSerializer;
        Fields: super::fields::GeneralFieldsSerializer;
        // prebuilt serializers are manually constructed, and thus manually added to the `CombinedSerializer` enum
        Prebuilt: super::prebuilt::PrebuiltSerializer;
        // polymorphism trampoline is manually constructed to wrap models and dataclasses with
        // polymorphic serialization
        PolymorphismTrampoline: super::polymorphism_trampoline::PolymorphismTrampoline;
    }
    // `find_only` is for type_serializers which are built directly via the `type` key and `find_serializer`
    // but aren't actually used for serialization, e.g. their `build` method must return another serializer
    find_only: {
        super::type_serializers::other::ChainBuilder;
        super::type_serializers::other::CustomErrorBuilder;
        super::type_serializers::other::CallBuilder;
        super::type_serializers::other::LaxOrStrictBuilder;
        super::type_serializers::other::ArgumentsBuilder;
        super::type_serializers::other::IsInstanceBuilder;
        super::type_serializers::other::IsSubclassBuilder;
        super::type_serializers::other::CallableBuilder;
        super::type_serializers::definitions::DefinitionsSerializerBuilder;
        super::type_serializers::dataclass::DataclassArgsBuilder;
        super::type_serializers::function::FunctionBeforeSerializerBuilder;
        super::type_serializers::function::FunctionAfterSerializerBuilder;
        super::type_serializers::function::FunctionPlainSerializerBuilder;
        super::type_serializers::function::FunctionWrapSerializerBuilder;
        super::type_serializers::model::ModelFieldsBuilder;
    }
    // `both` means the struct is added to both the `CombinedSerializer` enum and the match statement in
    // `find_serializer` so they can be used via a `type` str.
    both: {
        None: super::type_serializers::simple::NoneSerializer;
        Nullable: super::type_serializers::nullable::NullableSerializer;
        Int: super::type_serializers::simple::IntSerializer;
        Bool: super::type_serializers::simple::BoolSerializer;
        Float: super::type_serializers::float::FloatSerializer;
        Decimal: super::type_serializers::decimal::DecimalSerializer;
        Fraction: super::type_serializers::fraction::FractionSerializer;
        Str: super::type_serializers::string::StrSerializer;
        Bytes: super::type_serializers::bytes::BytesSerializer;
        Datetime: super::type_serializers::datetime_etc::DatetimeSerializer;
        TimeDelta: super::type_serializers::timedelta::TimeDeltaSerializer;
        Date: super::type_serializers::datetime_etc::DateSerializer;
        Time: super::type_serializers::datetime_etc::TimeSerializer;
        List: super::type_serializers::list::ListSerializer;
        Deque: super::type_serializers::deque::DequeSerializer;
        Set: super::type_serializers::set_frozenset::SetSerializer;
        FrozenSet: super::type_serializers::set_frozenset::FrozenSetSerializer;
        Generator: super::type_serializers::generator::GeneratorSerializer;
        Dict: super::type_serializers::dict::DictSerializer;
        FrozenDict: super::type_serializers::frozendict::FrozenDictSerializer;
        OrderedDict: super::type_serializers::ordered_dict::OrderedDictSerializer;
        Counter: super::type_serializers::counter::CounterSerializer;
        Model: super::type_serializers::model::ModelSerializer;
        Dataclass: super::type_serializers::dataclass::DataclassSerializer;
        Url: super::type_serializers::url::UrlSerializer;
        MultiHostUrl: super::type_serializers::url::MultiHostUrlSerializer;
        Uuid: super::type_serializers::uuid::UuidSerializer;
        Any: super::type_serializers::any::AnySerializer;
        Format: super::type_serializers::format::FormatSerializer;
        ToString: super::type_serializers::format::ToStringSerializer;
        WithDefault: super::type_serializers::with_default::WithDefaultSerializer;
        Json: super::type_serializers::json::JsonSerializer;
        JsonOrPython: super::type_serializers::json_or_python::JsonOrPythonSerializer;
        Union: super::type_serializers::union::UnionSerializer;
        TaggedUnion: super::type_serializers::union::TaggedUnionSerializer;
        Literal: super::type_serializers::literal::LiteralSerializer;
        MissingSentinel: super::type_serializers::missing_sentinel::MissingSentinelSerializer;
        Ellipsis: super::type_serializers::ellipsis::EllipsisSerializer;
        Enum: super::type_serializers::enum_::EnumSerializer;
        Recursive: super::type_serializers::definitions::DefinitionRefSerializer;
        Tuple: super::type_serializers::tuple::TupleSerializer;
        Complex: super::type_serializers::complex::ComplexSerializer;
        TypedDict: super::type_serializers::typed_dict::TypedDictSerializer;
        NamedTuple: super::type_serializers::named_tuple::NamedTupleSerializer;
    }
}

impl CombinedSerializer {
    fn _build(
        schema: &Bound<'_, PyDict>,
        config: Option<&Bound<'_, PyDict>>,
        definitions: &mut DefinitionsBuilder<Arc<CombinedSerializer>>,
        use_prebuilt: bool,
    ) -> PyResult<Arc<CombinedSerializer>> {
        let py = schema.py();
        let type_key = intern!(py, "type");

        if let Some(ser_schema) = schema.get_as::<Bound<'_, PyDict>>(intern!(py, "serialization"))? {
            let op_ser_type: Option<Bound<'_, PyString>> = ser_schema.get_as(type_key)?;
            match op_ser_type.as_ref().map(|py_str| py_str.to_str()).transpose()? {
                Some("function-plain") => {
                    // `function-plain` is a special case, not included in `find_serializer` since it means
                    // something different in `schema.type`
                    // NOTE! we use the `schema` here, not `ser_schema`
                    return super::type_serializers::function::FunctionPlainSerializer::build(
                        schema,
                        config,
                        definitions,
                    )
                    .map_err(|err| py_schema_error_type!("Error building `function-plain` serializer:\n  {err}"));
                }
                Some("function-wrap") => {
                    // `function-wrap` is also a special case, not included in `find_serializer` since it mean
                    // something different in `schema.type`
                    // NOTE! we use the `schema` here, not `ser_schema`
                    return super::type_serializers::function::FunctionWrapSerializer::build(
                        schema,
                        config,
                        definitions,
                    )
                    .map_err(|err| py_schema_error_type!("Error building `function-wrap` serializer:\n  {err}"));
                }
                Some(
                    // applies to lists tuples and dicts, does not override the main schema `type`
                    "include-exclude-sequence" | "include-exclude-dict"
                    // applies specifically to bytes, does not override the main schema `type`
                    | "base64"
                )
                // if `schema.serialization.type` is None, fall back to `schema.type`
                | None => (),
                Some(_) => {
                    // otherwise, `schema.serialization` is an arbitrary core schema (which includes the
                    // simple `{'type': ...}` ser schemas), so build a serializer from it as if it was
                    // the main schema (this ensures nested `serialization` schemas, prebuilt serializers
                    // and polymorphic serialization are handled). In this case, it's an error if a
                    // serializer isn't found.
                    // Note that as a consequence, `function-plain`/`function-wrap` *validator* schemas can't
                    // be used as `schema.serialization`, as they are interpreted as the function *ser* schemas.
                    return Self::build(&ser_schema, config, definitions);
                }
            }
        }

        let type_: Bound<'_, PyString> = schema.get_as_req(type_key)?;
        let type_ = type_.to_str()?;

        if use_prebuilt {
            // if we have a SchemaSerializer on the type already, use it
            if let Ok(Some(prebuilt_serializer)) =
                super::prebuilt::PrebuiltSerializer::try_get_from_schema(type_, schema)
            {
                return Ok(Arc::new(prebuilt_serializer));
            }
        }

        Self::find_serializer(type_, schema, config, definitions)
    }

    fn maybe_wrap_in_polymorphism_trampoline(
        serializer: Arc<CombinedSerializer>,
        schema: &Bound<'_, PyDict>,
    ) -> PyResult<Arc<CombinedSerializer>> {
        let py = schema.py();
        let type_: Bound<'_, PyString> = schema.get_as_req(intern!(py, "type"))?;
        let type_ = type_.to_str()?;

        // Note: it could make sense to generalize this behavior for any type that may have subclasses,
        // but apart from models and dataclasses, that would be for arbitrary types where custom serialization
        // has to be defined already.
        if type_ == "model" || type_ == "dataclass" {
            // Get polymorphic serialization from config
            let config = schema.get_as::<Bound<'_, PyDict>>(intern!(py, "config"))?;
            let polymorphic_serialization: bool = config
                .and_then(|cfg| cfg.get_as(intern!(py, "polymorphic_serialization")).transpose())
                .unwrap_or(Ok(false))?;

            // Unconditionally wrap in PolymorphismTrampoline, because runtime flag might still enable it
            Ok(Arc::new(
                PolymorphismTrampoline::new(
                    schema.get_as_req(intern!(py, "cls"))?,
                    serializer,
                    polymorphic_serialization,
                )
                .into(),
            ))
        } else {
            Ok(serializer)
        }
    }

    /// Main recursive way to call serializers, supports possible recursive type inference by
    /// switching to type inference mode eagerly.
    pub fn to_python<'py>(
        &self,
        value: &Bound<'py, PyAny>,
        state: &mut SerializationState<'py>,
    ) -> PyResult<Py<PyAny>> {
        if state.extra.serialize_as_any {
            infer_to_python(value, state)
        } else {
            self.to_python_no_infer(value, state)
        }
    }

    /// Variant of the above which does not fall back to inference mode immediately
    #[inline]
    pub fn to_python_no_infer<'py>(
        &self,
        value: &Bound<'py, PyAny>,
        state: &mut SerializationState<'py>,
    ) -> PyResult<Py<PyAny>> {
        TypeSerializer::to_python(self, value, state)
    }

    pub fn json_key<'a, 'py>(
        &self,
        key: &'a Bound<'py, PyAny>,
        state: &mut SerializationState<'py>,
    ) -> PyResult<Cow<'a, str>> {
        if state.extra.serialize_as_any {
            infer_json_key(key, state)
        } else {
            self.json_key_no_infer(key, state)
        }
    }

    #[inline]
    pub fn json_key_no_infer<'a, 'py>(
        &self,
        key: &'a Bound<'py, PyAny>,
        state: &mut SerializationState<'py>,
    ) -> PyResult<Cow<'a, str>> {
        TypeSerializer::json_key(self, key, state)
    }

    pub fn serde_serialize<'py, S: serde::ser::Serializer>(
        &self,
        value: &Bound<'py, PyAny>,
        serializer: S,
        state: &mut SerializationState<'py>,
    ) -> Result<S::Ok, S::Error> {
        if state.extra.serialize_as_any {
            infer_serialize(value, serializer, state)
        } else {
            self.serde_serialize_no_infer(value, serializer, state)
        }
    }

    #[inline]
    pub fn serde_serialize_no_infer<'py, S: serde::ser::Serializer>(
        &self,
        value: &Bound<'py, PyAny>,
        serializer: S,
        state: &mut SerializationState<'py>,
    ) -> Result<S::Ok, S::Error> {
        TypeSerializer::serde_serialize(self, value, serializer, state)
    }
}

impl BuildSerializer for CombinedSerializer {
    // this value is never used, it's just here to satisfy the trait
    const EXPECTED_TYPE: &'static str = "";

    fn build(
        schema: &Bound<'_, PyDict>,
        config: Option<&Bound<'_, PyDict>>,
        definitions: &mut DefinitionsBuilder<Arc<CombinedSerializer>>,
    ) -> PyResult<Arc<CombinedSerializer>> {
        // Read use_prebuilt from the definitions builder - this ensures all nested
        // serializers respect the same setting as the top-level build
        let use_prebuilt = definitions.use_prebuilt();
        let serializer = Self::_build(schema, config, definitions, use_prebuilt)?;
        Self::maybe_wrap_in_polymorphism_trampoline(serializer, schema)
    }
}

// Implemented by hand because `enum_dispatch` fails with a proc macro compile error =/
impl PyGcTraverse for CombinedSerializer {
    fn py_gc_traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            CombinedSerializer::Function(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::FunctionWrap(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Fields(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Prebuilt(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::None(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Nullable(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Int(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Bool(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Float(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Decimal(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Fraction(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Str(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Bytes(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Datetime(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::TimeDelta(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Date(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Time(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::List(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Deque(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Set(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::FrozenSet(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Generator(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Dict(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::FrozenDict(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::OrderedDict(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Counter(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Model(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Dataclass(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Url(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::MultiHostUrl(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Any(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Format(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::ToString(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::WithDefault(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Json(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::JsonOrPython(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Union(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::TaggedUnion(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Literal(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::MissingSentinel(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Ellipsis(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Enum(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Recursive(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Tuple(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Uuid(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::Complex(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::TypedDict(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::NamedTuple(inner) => inner.py_gc_traverse(visit),
            CombinedSerializer::PolymorphismTrampoline(inner) => inner.py_gc_traverse(visit),
        }
    }
}

#[enum_dispatch(CombinedSerializer)]
pub(crate) trait TypeSerializer: Send + Sync + Debug {
    fn to_python<'py>(&self, value: &Bound<'py, PyAny>, state: &mut SerializationState<'py>) -> PyResult<Py<PyAny>>;

    fn json_key<'a, 'py>(
        &self,
        key: &'a Bound<'py, PyAny>,
        state: &mut SerializationState<'py>,
    ) -> PyResult<Cow<'a, str>>;

    fn invalid_as_json_key<'a, 'py>(
        &self,
        key: &'a Bound<'py, PyAny>,
        state: &mut SerializationState<'py>,
        expected_type: &'static str,
    ) -> PyResult<Cow<'a, str>> {
        if key.is_none() {
            py_err!(PyTypeError; "`{expected_type}` not valid as object key")
        } else {
            state.warn_fallback_py(self.get_name(), key)?;
            infer_json_key(key, state)
        }
    }

    fn serde_serialize<'py, S: serde::ser::Serializer>(
        &self,
        value: &Bound<'py, PyAny>,
        serializer: S,
        state: &mut SerializationState<'py>,
    ) -> Result<S::Ok, S::Error>;

    fn get_name(&self) -> &str;

    /// Used by union serializers to decide if it's worth trying again while allowing subclasses
    fn retry_with_lax_check(&self) -> bool {
        false
    }

    fn get_default(&self, _py: Python) -> PyResult<Option<Py<PyAny>>> {
        Ok(None)
    }
}

pub(crate) struct PydanticSerializer<'slf, 'py> {
    value: &'slf Bound<'py, PyAny>,
    serializer: &'slf CombinedSerializer,
    /// RefCell to allow mutable access to the state during serialization, we expect it
    /// to only ever be borrowed mutably once at a time.
    state: RefCell<&'slf mut SerializationState<'py>>,
}

impl<'slf, 'py> PydanticSerializer<'slf, 'py> {
    pub(crate) fn new(
        value: &'slf Bound<'py, PyAny>,
        serializer: &'slf CombinedSerializer,
        state: &'slf mut SerializationState<'py>,
    ) -> Self {
        Self {
            value,
            serializer: if state.extra.serialize_as_any {
                AnySerializer::get()
            } else {
                serializer
            },
            state: RefCell::new(state),
        }
    }

    /// Same as above but will not fall back to type inference when `serialize_as_any` is set
    pub(crate) fn new_no_infer(
        value: &'slf Bound<'py, PyAny>,
        serializer: &'slf CombinedSerializer,
        state: &'slf mut SerializationState<'py>,
    ) -> Self {
        Self {
            value,
            serializer,
            state: RefCell::new(state),
        }
    }
}

impl Serialize for PydanticSerializer<'_, '_> {
    fn serialize<S: serde::ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // inference is handled in the constructor
        self.serializer
            .serde_serialize_no_infer(self.value, serializer, &mut self.state.borrow_mut())
    }
}

struct EscapeNonAsciiFormatter;

impl Formatter for EscapeNonAsciiFormatter {
    fn write_string_fragment<W: ?Sized + Write>(&mut self, writer: &mut W, fragment: &str) -> io::Result<()> {
        let mut input = fragment;

        while let Some((idx, non_ascii_char)) = input.chars().enumerate().find(|(_, c)| !c.is_ascii()) {
            if idx > 0 {
                // write all ascii characters before the non-ascii one
                let ascii_run = &input[..idx];
                writer.write_all(ascii_run.as_bytes()).unwrap();
            }

            let codepoint = non_ascii_char as u32;
            if codepoint < 0xFFFF {
                // write basic codepoint as single escape
                write!(writer, "\\u{codepoint:04x}").unwrap();
            } else {
                // encode extended plane character as utf16 pair
                for escape in non_ascii_char.encode_utf16(&mut [0; 2]) {
                    write!(writer, "\\u{escape:04x}").unwrap();
                }
            }

            input = &input[(idx + non_ascii_char.len_utf8())..];
        }

        // write any ascii trailer
        writer.write_all(input.as_bytes())?;
        Ok(())
    }
}

/// JSON string content which is already fully escaped (quotes, backslashes and control
/// characters), e.g. containing lone surrogate escapes like `\udcff`. It can't go through
/// `serialize_str`, as the normal escaping would escape the backslashes a second time.
pub(crate) struct RawJsonEscaped(Vec<u8>);

impl Serialize for RawJsonEscaped {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // `PythonSerializer::serialize_bytes` writes the bytes verbatim, surrounded by
        // quotes (see `ser.rs`). Python `bytes` never reach this method.
        serializer.serialize_bytes(&self.0)
    }
}

impl RawJsonEscaped {
    /// Escape a Python string to JSON string content bytes, escaping lone surrogates like
    /// `\udcXX`, the same way as Python's `json.dumps` does.
    ///
    /// This is only intended for strings which can't be converted to Rust `&str`, i.e.
    /// strings containing lone surrogates — e.g. paths decoded with `surrogateescape`.
    ///
    /// The output contains no surrounding quotes, but otherwise the *full* escaping
    /// (quotes, backslashes, control characters) is performed, with non-ASCII characters
    /// left as-is; they are handled by the `Formatter` when the content is written
    /// (e.g. escaped for `ensure_ascii = true`).
    fn new(py_str: &Bound<'_, PyString>) -> PyResult<Self> {
        const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

        let encoded = {
            let ptr = unsafe {
                pyo3::ffi::PyUnicode_AsEncodedString(py_str.as_ptr(), c"utf-8".as_ptr(), c"surrogatepass".as_ptr())
            };
            let bound = unsafe { Bound::from_owned_ptr_or_err(py_str.py(), ptr)? };
            bound.cast::<PyBytes>()?.as_bytes().to_vec()
        };

        let mut out = Vec::with_capacity(encoded.len());
        let mut i = 0;
        while i < encoded.len() {
            match encoded[i] {
                b'\\' => {
                    out.extend_from_slice(b"\\\\");
                    i += 1;
                }
                b'"' => {
                    out.extend_from_slice(b"\\\"");
                    i += 1;
                }
                b'\x08' => {
                    out.extend_from_slice(b"\\b");
                    i += 1;
                }
                b'\x09' => {
                    out.extend_from_slice(b"\\t");
                    i += 1;
                }
                b'\x0A' => {
                    out.extend_from_slice(b"\\n");
                    i += 1;
                }
                b'\x0C' => {
                    out.extend_from_slice(b"\\f");
                    i += 1;
                }
                b'\x0D' => {
                    out.extend_from_slice(b"\\r");
                    i += 1;
                }
                b if b < 0x20 => {
                    // other control characters, e.g. `\u0007`
                    out.extend_from_slice(b"\\u00");
                    out.push(HEX_DIGITS[(b >> 4) as usize]);
                    out.push(HEX_DIGITS[(b & 0x0F) as usize]);
                    i += 1;
                }
                // lone surrogate, encoded by `surrogatepass` as `ED Bx xx`
                0xED if encoded.len() >= i + 3
                    && (0xA0..=0xBF).contains(&encoded[i + 1])
                    && (0x80..=0xBF).contains(&encoded[i + 2]) =>
                {
                    let codepoint =
                        0xD800u32 + (u32::from(encoded[i + 1] - 0xA0) << 6) + u32::from(encoded[i + 2] - 0x80);
                    out.extend_from_slice(b"\\u");
                    out.extend_from_slice(format!("{codepoint:04x}").as_bytes());
                    i += 3;
                }
                _ => {
                    // ASCII characters above `0x1F` (including `\u007F`/DEL, which serde_json
                    // leaves as-is) and all non-surrogate multi-byte characters, written raw
                    out.push(encoded[i]);
                    i += 1;
                }
            }
        }

        Ok(Self(out))
    }
}

struct EscapeNonAsciiPrettyFormatter<'a> {
    pretty: PrettyFormatter<'a>,
    escape_non_ascii: EscapeNonAsciiFormatter,
}

impl<'a> EscapeNonAsciiPrettyFormatter<'a> {
    pub fn with_indent(indent: &'a [u8]) -> Self {
        Self {
            pretty: PrettyFormatter::with_indent(indent),
            escape_non_ascii: EscapeNonAsciiFormatter,
        }
    }
}

macro_rules! defer {
    ($formatter:ident, $fun:ident) => {
        fn $fun<W>(&mut self, writer: &mut W) -> io::Result<()>
        where
            W: ?Sized + io::Write,
        {
            self.$formatter.$fun(writer)
        }
    };
    ($formatter:ident, $fun:ident, $val:ty) => {
        fn $fun<W>(&mut self, writer: &mut W, val: $val) -> io::Result<()>
        where
            W: ?Sized + io::Write,
        {
            self.$formatter.$fun(writer, val)
        }
    };
}

#[allow(clippy::needless_lifetimes)]
impl Formatter for EscapeNonAsciiPrettyFormatter<'_> {
    defer!(escape_non_ascii, write_string_fragment, &str);
    defer!(pretty, begin_array);
    defer!(pretty, end_array);
    defer!(pretty, begin_array_value, bool);
    defer!(pretty, end_array_value);
    defer!(pretty, begin_object);
    defer!(pretty, end_object);
    defer!(pretty, begin_object_key, bool);
    defer!(pretty, end_object_key);
    defer!(pretty, begin_object_value);
    defer!(pretty, end_object_value);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn to_json_bytes<'py>(
    value: &Bound<'py, PyAny>,
    serializer: &CombinedSerializer,
    state: &mut SerializationState<'py>,
    indent: Option<usize>,
    ensure_ascii: bool,
    expected_json_size: usize,
) -> PyResult<Vec<u8>> {
    let serializer = PydanticSerializer::new(value, serializer, state);

    let writer: Vec<u8> = Vec::with_capacity(expected_json_size);

    let bytes = match (indent, ensure_ascii) {
        (Some(indent), true) => {
            let indent = vec![b' '; indent];
            let formatter = EscapeNonAsciiPrettyFormatter::with_indent(&indent);
            let mut ser = PythonSerializer::with_formatter(writer, formatter);
            serializer.serialize(&mut ser).map_err(se_err_py_err)?;
            ser.into_inner()
        }
        (Some(indent), false) => {
            let indent = vec![b' '; indent];
            let formatter = PrettyFormatter::with_indent(&indent);
            let mut ser = PythonSerializer::with_formatter(writer, formatter);
            serializer.serialize(&mut ser).map_err(se_err_py_err)?;
            ser.into_inner()
        }
        (None, true) => {
            let mut ser = PythonSerializer::with_formatter(writer, EscapeNonAsciiFormatter);
            serializer.serialize(&mut ser).map_err(se_err_py_err)?;
            ser.into_inner()
        }
        (None, false) => {
            let mut ser = PythonSerializer::new(writer);
            serializer.serialize(&mut ser).map_err(se_err_py_err)?;
            ser.into_inner()
        }
    };

    Ok(bytes)
}

/// Common interface for doing serialization
pub(crate) trait DoSerialize {
    type Ok;
    type Error: From<PyErr>;

    fn serialize_no_infer<'py>(
        self,
        serializer: &CombinedSerializer,
        value: &Bound<'py, PyAny>,
        state: &mut SerializationState<'py>,
    ) -> Result<Self::Ok, Self::Error>;

    fn serialize_fallback<'py>(
        self,
        name: &str,
        value: &Bound<'py, PyAny>,
        state: &mut SerializationState<'py>,
    ) -> Result<Self::Ok, Self::Error>;

    fn serialize_str(self, value: &Bound<'_, PyString>) -> Result<Self::Ok, Self::Error>;

    type Map: SerializeMap<Ok = Self::Ok, Error = Self::Error>;

    fn serialize_map(self) -> Result<Self::Map, Self::Error>;
}

pub(crate) trait SerializeMap {
    type Ok;
    type Error: From<PyErr>;

    /// Serialize a key-value entry into the map
    fn serialize_entry<'py>(
        &mut self,
        key: &Bound<'py, PyAny>,
        key_serializer: &CombinedSerializer,
        value: &Bound<'py, PyAny>,
        value_serializer: &CombinedSerializer,
        state: &mut SerializationState<'py>,
    ) -> Result<(), Self::Error>;

    /// Serialize a string key-value entry into the map (avoids needing to "serialize" the key)
    fn serialize_entry_string_key<'py>(
        &mut self,
        key: &PyBackedStr,
        value: &Bound<'py, PyAny>,
        value_serializer: &CombinedSerializer,
        state: &mut SerializationState<'py>,
    ) -> Result<(), Self::Error>;

    fn end(self) -> Result<Self::Ok, Self::Error>;
}

/// Helper to create a `SerializeToPython` instance
pub fn serialize_to_python(py: Python<'_>) -> SerializeToPython<'_> {
    SerializeToPython(py)
}

/// Helper to create a `SerializeToJson` instance
pub fn serialize_to_json<S>(serializer: S) -> SerializeToJson<S> {
    SerializeToJson { serializer }
}

pub struct SerializeToPython<'py>(Python<'py>);

impl<'s> DoSerialize for SerializeToPython<'s> {
    type Ok = Py<PyAny>;
    type Error = PyErr;

    fn serialize_no_infer<'py>(
        self,
        serializer: &CombinedSerializer,
        value: &Bound<'py, PyAny>,
        state: &mut SerializationState<'py>,
    ) -> PyResult<Py<PyAny>> {
        serializer.to_python_no_infer(value, state)
    }

    fn serialize_fallback<'py>(
        self,
        name: &str,
        value: &Bound<'py, PyAny>,
        state: &mut SerializationState<'py>,
    ) -> PyResult<Py<PyAny>> {
        state.warn_fallback_py(name, value)?;
        infer_to_python(value, state)
    }

    fn serialize_str(self, value: &Bound<'_, PyString>) -> Result<Py<PyAny>, PyErr> {
        value.into_py_any(value.py())
    }

    type Map = Bound<'s, PyDict>;

    fn serialize_map(self) -> Result<Bound<'s, PyDict>, PyErr> {
        Ok(PyDict::new(self.0))
    }
}

impl SerializeMap for Bound<'_, PyDict> {
    type Ok = Py<PyAny>;
    type Error = PyErr;

    fn serialize_entry<'py>(
        &mut self,
        key: &Bound<'py, PyAny>,
        key_serializer: &CombinedSerializer,
        value: &Bound<'py, PyAny>,
        value_serializer: &CombinedSerializer,
        state: &mut SerializationState<'py>,
    ) -> Result<(), Self::Error> {
        if matches!(state.extra.mode, SerMode::Json) {
            let key = key_serializer.json_key_no_infer(key, state)?;
            let py_value = value_serializer.to_python_no_infer(value, state)?;
            self.set_item(key, py_value)?;
        } else {
            let py_key = key_serializer.to_python_no_infer(key, state)?;
            let py_value = value_serializer.to_python_no_infer(value, state)?;
            self.set_item(py_key, py_value)?;
        }
        Ok(())
    }

    fn serialize_entry_string_key<'py>(
        &mut self,
        key: &PyBackedStr,
        value: &Bound<'py, PyAny>,
        value_serializer: &CombinedSerializer,
        state: &mut SerializationState<'py>,
    ) -> Result<(), Self::Error> {
        let py_value = value_serializer.to_python_no_infer(value, state)?;
        self.set_item(key, py_value)?;
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(self.into_any().unbind())
    }
}

pub(crate) struct SerializeToJson<S> {
    serializer: S,
}

impl<S: Serializer> DoSerialize for SerializeToJson<S> {
    type Ok = S::Ok;
    type Error = WrappedSerError<S::Error>;

    fn serialize_no_infer<'py>(
        self,
        serializer: &CombinedSerializer,
        value: &Bound<'py, PyAny>,
        state: &mut SerializationState<'py>,
    ) -> Result<S::Ok, WrappedSerError<S::Error>> {
        serializer
            .serde_serialize_no_infer(value, self.serializer, state)
            .map_err(WrappedSerError)
    }

    fn serialize_fallback<'py>(
        self,
        name: &str,
        value: &Bound<'py, PyAny>,
        state: &mut SerializationState<'py>,
    ) -> Result<S::Ok, WrappedSerError<S::Error>> {
        state.warn_fallback_ser::<S>(name, value).map_err(WrappedSerError)?;
        infer_serialize(value, self.serializer, state).map_err(WrappedSerError)
    }

    fn serialize_str(self, value: &Bound<'_, PyString>) -> Result<S::Ok, WrappedSerError<S::Error>> {
        match value.to_str() {
            Ok(s) => self.serializer.serialize_str(s).map_err(WrappedSerError),
            Err(_) => {
                // the string contains lone surrogates (e.g. produced by `surrogateescape`
                // for filesystem paths on Linux), which can't be represented as Rust `&str`;
                // escape them as `\udcXX` like `json.dumps` does, and write the fully-escaped
                // content raw
                RawJsonEscaped::new(value)?
                    .serialize(self.serializer)
                    .map_err(WrappedSerError)
            }
        }
    }

    type Map = SerdeMapSerializer<S>;

    fn serialize_map(self) -> Result<SerdeMapSerializer<S>, Self::Error> {
        Ok(SerdeMapSerializer::<S> {
            // choice not to pass size_hint here is because JSON maps don't need a size hint
            map: self.serializer.serialize_map(None).map_err(WrappedSerError)?,
        })
    }
}

pub(crate) struct SerdeMapSerializer<S: Serializer> {
    map: S::SerializeMap,
}

impl<S: Serializer> SerializeMap for SerdeMapSerializer<S> {
    type Ok = S::Ok;
    type Error = WrappedSerError<S::Error>;

    fn serialize_entry<'py>(
        &mut self,
        key: &Bound<'py, PyAny>,
        key_serializer: &CombinedSerializer,
        value: &Bound<'py, PyAny>,
        value_serializer: &CombinedSerializer,
        state: &mut SerializationState<'py>,
    ) -> Result<(), Self::Error> {
        let key = key_serializer.json_key_no_infer(key, state)?;
        let value_ser = PydanticSerializer::new_no_infer(value, value_serializer, state);
        self.map.serialize_entry(&key, &value_ser).map_err(WrappedSerError)
    }

    fn serialize_entry_string_key<'py>(
        &mut self,
        key: &PyBackedStr,
        value: &Bound<'py, PyAny>,
        value_serializer: &CombinedSerializer,
        state: &mut SerializationState<'py>,
    ) -> Result<(), Self::Error> {
        let value_ser = PydanticSerializer::new_no_infer(value, value_serializer, state);
        self.map.serialize_entry(&**key, &value_ser).map_err(WrappedSerError)
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.map.end().map_err(WrappedSerError)
    }
}
