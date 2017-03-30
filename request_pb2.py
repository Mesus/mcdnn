# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: request.proto

from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)




DESCRIPTOR = _descriptor.FileDescriptor(
  name='request.proto',
  package='uw.syhan.mcdnn',
  serialized_pb='\n\rrequest.proto\x12\x0euw.syhan.mcdnn\"c\n\nDNNRequest\x12)\n\x04type\x18\x01 \x02(\x0e\x32\x1b.uw.syhan.mcdnn.RequestType\x12\r\n\x05layer\x18\x02 \x01(\x05\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\x12\r\n\x05model\x18\x04 \x01(\t\"g\n\x0b\x44NNResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07latency\x18\x02 \x01(\x01\x12\x0e\n\x06result\x18\x03 \x01(\x05\x12\x12\n\nresult_str\x18\x04 \x01(\t\x12\x12\n\nconfidence\x18\x05 \x01(\x01*.\n\x0bRequestType\x12\x08\n\x04\x46\x41\x43\x45\x10\x01\x12\n\n\x06OBJECT\x10\x02\x12\t\n\x05SCENE\x10\x03')

_REQUESTTYPE = _descriptor.EnumDescriptor(
  name='RequestType',
  full_name='uw.syhan.mcdnn.RequestType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='FACE', index=0, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBJECT', index=1, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SCENE', index=2, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=239,
  serialized_end=285,
)

RequestType = enum_type_wrapper.EnumTypeWrapper(_REQUESTTYPE)
FACE = 1
OBJECT = 2
SCENE = 3



_DNNREQUEST = _descriptor.Descriptor(
  name='DNNRequest',
  full_name='uw.syhan.mcdnn.DNNRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='uw.syhan.mcdnn.DNNRequest.type', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='layer', full_name='uw.syhan.mcdnn.DNNRequest.layer', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data', full_name='uw.syhan.mcdnn.DNNRequest.data', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value="",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='model', full_name='uw.syhan.mcdnn.DNNRequest.model', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=33,
  serialized_end=132,
)


_DNNRESPONSE = _descriptor.Descriptor(
  name='DNNResponse',
  full_name='uw.syhan.mcdnn.DNNResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='uw.syhan.mcdnn.DNNResponse.success', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='latency', full_name='uw.syhan.mcdnn.DNNResponse.latency', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='result', full_name='uw.syhan.mcdnn.DNNResponse.result', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='result_str', full_name='uw.syhan.mcdnn.DNNResponse.result_str', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='confidence', full_name='uw.syhan.mcdnn.DNNResponse.confidence', index=4,
      number=5, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=134,
  serialized_end=237,
)

_DNNREQUEST.fields_by_name['type'].enum_type = _REQUESTTYPE
DESCRIPTOR.message_types_by_name['DNNRequest'] = _DNNREQUEST
DESCRIPTOR.message_types_by_name['DNNResponse'] = _DNNRESPONSE

class DNNRequest(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _DNNREQUEST

  # @@protoc_insertion_point(class_scope:uw.syhan.mcdnn.DNNRequest)

class DNNResponse(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _DNNRESPONSE

  # @@protoc_insertion_point(class_scope:uw.syhan.mcdnn.DNNResponse)


# @@protoc_insertion_point(module_scope)
