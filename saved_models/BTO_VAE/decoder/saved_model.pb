??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
?
conv1d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameconv1d_transpose/kernel
?
+conv1d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose/kernel*#
_output_shapes
:?*
dtype0
?
conv1d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv1d_transpose/bias
{
)conv1d_transpose/bias/Read/ReadVariableOpReadVariableOpconv1d_transpose/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
 
h


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api


0
1
2
3
 


0
1
2
3
?
non_trainable_variables
layer_metrics
layer_regularization_losses
trainable_variables
metrics
regularization_losses

layers
	variables
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1


0
1
 
?
non_trainable_variables
 layer_metrics
!layer_regularization_losses
trainable_variables
	variables
"metrics

#layers
regularization_losses
 
 
 
?
$non_trainable_variables
%layer_metrics
&layer_regularization_losses
trainable_variables
	variables
'metrics

(layers
regularization_losses
ca
VARIABLE_VALUEconv1d_transpose/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv1d_transpose/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
)non_trainable_variables
*layer_metrics
+layer_regularization_losses
trainable_variables
	variables
,metrics

-layers
regularization_losses
 
 
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasconv1d_transpose/kernelconv1d_transpose/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *-
f(R&
$__inference_signature_wrapper_160125
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp+conv1d_transpose/kernel/Read/ReadVariableOp)conv1d_transpose/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *(
f#R!
__inference__traced_save_160338
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasconv1d_transpose/kernelconv1d_transpose/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *+
f&R$
"__inference__traced_restore_160360??
?
?
,__inference_CNN_decoder_layer_call_fn_160010
input_1
unknown:	?
	unknown_0:	? 
	unknown_1:?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_1599992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
,__inference_CNN_decoder_layer_call_fn_160151

inputs
unknown:	?
	unknown_0:	? 
	unknown_1:?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_1600562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_160095
input_1
dense_160083:	?
dense_160085:	?.
conv1d_transpose_160089:?%
conv1d_transpose_160091:
identity??(conv1d_transpose/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_160083dense_160085*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1599722
dense/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1599912
reshape/PartitionedCall?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_transpose_160089conv1d_transpose_160091*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_1599442*
(conv1d_transpose/StatefulPartitionedCall?
IdentityIdentity1conv1d_transpose/StatefulPartitionedCall:output:0)^conv1d_transpose/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
A__inference_dense_layer_call_and_return_conditional_losses_160285

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_159999

inputs
dense_159973:	?
dense_159975:	?.
conv1d_transpose_159993:?%
conv1d_transpose_159995:
identity??(conv1d_transpose/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_159973dense_159975*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1599722
dense/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1599912
reshape/PartitionedCall?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_transpose_159993conv1d_transpose_159995*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_1599442*
(conv1d_transpose/StatefulPartitionedCall?
IdentityIdentity1conv1d_transpose/StatefulPartitionedCall:output:0)^conv1d_transpose/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?2
?
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_159944

inputsL
5conv1d_transpose_expanddims_1_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulQ
add/yConst*
_output_shapes
: *
dtype0*
value
B :?2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2w
stackPackstrided_slice:output:0add:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack?
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv1d_transpose/ExpandDims?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp?
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2
conv1d_transpose/ExpandDims_1?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1?
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
2
conv1d_transpose?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
2
conv1d_transpose/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
1__inference_conv1d_transpose_layer_call_fn_159954

inputs
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_1599442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
,__inference_CNN_decoder_layer_call_fn_160138

inputs
unknown:	?
	unknown_0:	? 
	unknown_1:?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_1599992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_dense_layer_call_and_return_conditional_losses_159972

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_159991

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:??????????2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_160110
input_1
dense_160098:	?
dense_160100:	?.
conv1d_transpose_160104:?%
conv1d_transpose_160106:
identity??(conv1d_transpose/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_160098dense_160100*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1599722
dense/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1599912
reshape/PartitionedCall?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_transpose_160104conv1d_transpose_160106*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_1599442*
(conv1d_transpose/StatefulPartitionedCall?
IdentityIdentity1conv1d_transpose/StatefulPartitionedCall:output:0)^conv1d_transpose/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
D
(__inference_reshape_layer_call_fn_160290

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1599912
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_dense_layer_call_fn_160274

inputs
unknown:	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1599722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?e
?
!__inference__wrapped_model_159902
input_1C
0cnn_decoder_dense_matmul_readvariableop_resource:	?@
1cnn_decoder_dense_biasadd_readvariableop_resource:	?i
Rcnn_decoder_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:?J
<cnn_decoder_conv1d_transpose_biasadd_readvariableop_resource:
identity??3CNN_decoder/conv1d_transpose/BiasAdd/ReadVariableOp?ICNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?(CNN_decoder/dense/BiasAdd/ReadVariableOp?'CNN_decoder/dense/MatMul/ReadVariableOp?
'CNN_decoder/dense/MatMul/ReadVariableOpReadVariableOp0cnn_decoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'CNN_decoder/dense/MatMul/ReadVariableOp?
CNN_decoder/dense/MatMulMatMulinput_1/CNN_decoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
CNN_decoder/dense/MatMul?
(CNN_decoder/dense/BiasAdd/ReadVariableOpReadVariableOp1cnn_decoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(CNN_decoder/dense/BiasAdd/ReadVariableOp?
CNN_decoder/dense/BiasAddBiasAdd"CNN_decoder/dense/MatMul:product:00CNN_decoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
CNN_decoder/dense/BiasAdd?
CNN_decoder/dense/ReluRelu"CNN_decoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
CNN_decoder/dense/Relu?
CNN_decoder/reshape/ShapeShape$CNN_decoder/dense/Relu:activations:0*
T0*
_output_shapes
:2
CNN_decoder/reshape/Shape?
'CNN_decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'CNN_decoder/reshape/strided_slice/stack?
)CNN_decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)CNN_decoder/reshape/strided_slice/stack_1?
)CNN_decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)CNN_decoder/reshape/strided_slice/stack_2?
!CNN_decoder/reshape/strided_sliceStridedSlice"CNN_decoder/reshape/Shape:output:00CNN_decoder/reshape/strided_slice/stack:output:02CNN_decoder/reshape/strided_slice/stack_1:output:02CNN_decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!CNN_decoder/reshape/strided_slice?
#CNN_decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#CNN_decoder/reshape/Reshape/shape/1?
#CNN_decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#CNN_decoder/reshape/Reshape/shape/2?
!CNN_decoder/reshape/Reshape/shapePack*CNN_decoder/reshape/strided_slice:output:0,CNN_decoder/reshape/Reshape/shape/1:output:0,CNN_decoder/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!CNN_decoder/reshape/Reshape/shape?
CNN_decoder/reshape/ReshapeReshape$CNN_decoder/dense/Relu:activations:0*CNN_decoder/reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
CNN_decoder/reshape/Reshape?
"CNN_decoder/conv1d_transpose/ShapeShape$CNN_decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:2$
"CNN_decoder/conv1d_transpose/Shape?
0CNN_decoder/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0CNN_decoder/conv1d_transpose/strided_slice/stack?
2CNN_decoder/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2CNN_decoder/conv1d_transpose/strided_slice/stack_1?
2CNN_decoder/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2CNN_decoder/conv1d_transpose/strided_slice/stack_2?
*CNN_decoder/conv1d_transpose/strided_sliceStridedSlice+CNN_decoder/conv1d_transpose/Shape:output:09CNN_decoder/conv1d_transpose/strided_slice/stack:output:0;CNN_decoder/conv1d_transpose/strided_slice/stack_1:output:0;CNN_decoder/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*CNN_decoder/conv1d_transpose/strided_slice?
2CNN_decoder/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2CNN_decoder/conv1d_transpose/strided_slice_1/stack?
4CNN_decoder/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4CNN_decoder/conv1d_transpose/strided_slice_1/stack_1?
4CNN_decoder/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4CNN_decoder/conv1d_transpose/strided_slice_1/stack_2?
,CNN_decoder/conv1d_transpose/strided_slice_1StridedSlice+CNN_decoder/conv1d_transpose/Shape:output:0;CNN_decoder/conv1d_transpose/strided_slice_1/stack:output:0=CNN_decoder/conv1d_transpose/strided_slice_1/stack_1:output:0=CNN_decoder/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,CNN_decoder/conv1d_transpose/strided_slice_1?
"CNN_decoder/conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"CNN_decoder/conv1d_transpose/mul/y?
 CNN_decoder/conv1d_transpose/mulMul5CNN_decoder/conv1d_transpose/strided_slice_1:output:0+CNN_decoder/conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2"
 CNN_decoder/conv1d_transpose/mul?
"CNN_decoder/conv1d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"CNN_decoder/conv1d_transpose/add/y?
 CNN_decoder/conv1d_transpose/addAddV2$CNN_decoder/conv1d_transpose/mul:z:0+CNN_decoder/conv1d_transpose/add/y:output:0*
T0*
_output_shapes
: 2"
 CNN_decoder/conv1d_transpose/add?
$CNN_decoder/conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$CNN_decoder/conv1d_transpose/stack/2?
"CNN_decoder/conv1d_transpose/stackPack3CNN_decoder/conv1d_transpose/strided_slice:output:0$CNN_decoder/conv1d_transpose/add:z:0-CNN_decoder/conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:2$
"CNN_decoder/conv1d_transpose/stack?
<CNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2>
<CNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims/dim?
8CNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDims$CNN_decoder/reshape/Reshape:output:0ECNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2:
8CNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims?
ICNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpRcnn_decoder_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02K
ICNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?
>CNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2@
>CNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/dim?
:CNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsQCNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0GCNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2<
:CNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1?
ACNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
ACNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack?
CCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
CCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_1?
CCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
CCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_2?
;CNN_decoder/conv1d_transpose/conv1d_transpose/strided_sliceStridedSlice+CNN_decoder/conv1d_transpose/stack:output:0JCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0LCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0LCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2=
;CNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice?
CCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
CCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack?
ECNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
ECNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1?
ECNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
ECNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2?
=CNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1StridedSlice+CNN_decoder/conv1d_transpose/stack:output:0LCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0NCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0NCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2?
=CNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1?
=CNN_decoder/conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=CNN_decoder/conv1d_transpose/conv1d_transpose/concat/values_1?
9CNN_decoder/conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9CNN_decoder/conv1d_transpose/conv1d_transpose/concat/axis?
4CNN_decoder/conv1d_transpose/conv1d_transpose/concatConcatV2DCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice:output:0FCNN_decoder/conv1d_transpose/conv1d_transpose/concat/values_1:output:0FCNN_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1:output:0BCNN_decoder/conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:26
4CNN_decoder/conv1d_transpose/conv1d_transpose/concat?
-CNN_decoder/conv1d_transpose/conv1d_transposeConv2DBackpropInput=CNN_decoder/conv1d_transpose/conv1d_transpose/concat:output:0CCNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1:output:0ACNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
2/
-CNN_decoder/conv1d_transpose/conv1d_transpose?
5CNN_decoder/conv1d_transpose/conv1d_transpose/SqueezeSqueeze6CNN_decoder/conv1d_transpose/conv1d_transpose:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
27
5CNN_decoder/conv1d_transpose/conv1d_transpose/Squeeze?
3CNN_decoder/conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOp<cnn_decoder_conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3CNN_decoder/conv1d_transpose/BiasAdd/ReadVariableOp?
$CNN_decoder/conv1d_transpose/BiasAddBiasAdd>CNN_decoder/conv1d_transpose/conv1d_transpose/Squeeze:output:0;CNN_decoder/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2&
$CNN_decoder/conv1d_transpose/BiasAdd?
IdentityIdentity-CNN_decoder/conv1d_transpose/BiasAdd:output:04^CNN_decoder/conv1d_transpose/BiasAdd/ReadVariableOpJ^CNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp)^CNN_decoder/dense/BiasAdd/ReadVariableOp(^CNN_decoder/dense/MatMul/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2j
3CNN_decoder/conv1d_transpose/BiasAdd/ReadVariableOp3CNN_decoder/conv1d_transpose/BiasAdd/ReadVariableOp2?
ICNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpICNN_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2T
(CNN_decoder/dense/BiasAdd/ReadVariableOp(CNN_decoder/dense/BiasAdd/ReadVariableOp2R
'CNN_decoder/dense/MatMul/ReadVariableOp'CNN_decoder/dense/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_160056

inputs
dense_160044:	?
dense_160046:	?.
conv1d_transpose_160050:?%
conv1d_transpose_160052:
identity??(conv1d_transpose/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_160044dense_160046*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1599722
dense/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1599912
reshape/PartitionedCall?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_transpose_160050conv1d_transpose_160052*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_1599442*
(conv1d_transpose/StatefulPartitionedCall?
IdentityIdentity1conv1d_transpose/StatefulPartitionedCall:output:0)^conv1d_transpose/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?U
?
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_160208

inputs7
$dense_matmul_readvariableop_resource:	?4
%dense_biasadd_readvariableop_resource:	?]
Fconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:?>
0conv1d_transpose_biasadd_readvariableop_resource:
identity??'conv1d_transpose/BiasAdd/ReadVariableOp?=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Reluf
reshape/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_sliceu
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense/Relu:activations:0reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
reshape/Reshapex
conv1d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose/Shape?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/Shape:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/Shape:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose/strided_slice_1r
conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose/mul/y?
conv1d_transpose/mulMul)conv1d_transpose/strided_slice_1:output:0conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose/muls
conv1d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value
B :?2
conv1d_transpose/add/y?
conv1d_transpose/addAddV2conv1d_transpose/mul:z:0conv1d_transpose/add/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose/addv
conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose/stack/2?
conv1d_transpose/stackPack'conv1d_transpose/strided_slice:output:0conv1d_transpose/add:z:0!conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/stack?
0conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0conv1d_transpose/conv1d_transpose/ExpandDims/dim?
,conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDimsreshape/Reshape:output:09conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2.
,conv1d_transpose/conv1d_transpose/ExpandDims?
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpFconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02?
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dim?
.conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsEconv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0;conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?20
.conv1d_transpose/conv1d_transpose/ExpandDims_1?
5conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5conv1d_transpose/conv1d_transpose/strided_slice/stack?
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1?
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2?
/conv1d_transpose/conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/stack:output:0>conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask21
/conv1d_transpose/conv1d_transpose/strided_slice?
7conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice_1/stack?
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1?
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2?
1conv1d_transpose/conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask23
1conv1d_transpose/conv1d_transpose/strided_slice_1?
1conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:23
1conv1d_transpose/conv1d_transpose/concat/values_1?
-conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-conv1d_transpose/conv1d_transpose/concat/axis?
(conv1d_transpose/conv1d_transpose/concatConcatV28conv1d_transpose/conv1d_transpose/strided_slice:output:0:conv1d_transpose/conv1d_transpose/concat/values_1:output:0:conv1d_transpose/conv1d_transpose/strided_slice_1:output:06conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(conv1d_transpose/conv1d_transpose/concat?
!conv1d_transpose/conv1d_transposeConv2DBackpropInput1conv1d_transpose/conv1d_transpose/concat:output:07conv1d_transpose/conv1d_transpose/ExpandDims_1:output:05conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
2#
!conv1d_transpose/conv1d_transpose?
)conv1d_transpose/conv1d_transpose/SqueezeSqueeze*conv1d_transpose/conv1d_transpose:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2+
)conv1d_transpose/conv1d_transpose/Squeeze?
'conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv1d_transpose/BiasAdd/ReadVariableOp?
conv1d_transpose/BiasAddBiasAdd2conv1d_transpose/conv1d_transpose/Squeeze:output:0/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
conv1d_transpose/BiasAdd?
IdentityIdentity!conv1d_transpose/BiasAdd:output:0(^conv1d_transpose/BiasAdd/ReadVariableOp>^conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2R
'conv1d_transpose/BiasAdd/ReadVariableOp'conv1d_transpose/BiasAdd/ReadVariableOp2~
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_CNN_decoder_layer_call_fn_160080
input_1
unknown:	?
	unknown_0:	? 
	unknown_1:?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_1600562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?U
?
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_160265

inputs7
$dense_matmul_readvariableop_resource:	?4
%dense_biasadd_readvariableop_resource:	?]
Fconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:?>
0conv1d_transpose_biasadd_readvariableop_resource:
identity??'conv1d_transpose/BiasAdd/ReadVariableOp?=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Reluf
reshape/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_sliceu
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense/Relu:activations:0reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
reshape/Reshapex
conv1d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose/Shape?
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack?
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1?
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2?
conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/Shape:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv1d_transpose/strided_slice?
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack?
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_1?
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2?
 conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/Shape:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose/strided_slice_1r
conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose/mul/y?
conv1d_transpose/mulMul)conv1d_transpose/strided_slice_1:output:0conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose/muls
conv1d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value
B :?2
conv1d_transpose/add/y?
conv1d_transpose/addAddV2conv1d_transpose/mul:z:0conv1d_transpose/add/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose/addv
conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose/stack/2?
conv1d_transpose/stackPack'conv1d_transpose/strided_slice:output:0conv1d_transpose/add:z:0!conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/stack?
0conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0conv1d_transpose/conv1d_transpose/ExpandDims/dim?
,conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDimsreshape/Reshape:output:09conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2.
,conv1d_transpose/conv1d_transpose/ExpandDims?
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpFconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02?
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dim?
.conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsEconv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0;conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?20
.conv1d_transpose/conv1d_transpose/ExpandDims_1?
5conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5conv1d_transpose/conv1d_transpose/strided_slice/stack?
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1?
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2?
/conv1d_transpose/conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/stack:output:0>conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask21
/conv1d_transpose/conv1d_transpose/strided_slice?
7conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice_1/stack?
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1?
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2?
1conv1d_transpose/conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask23
1conv1d_transpose/conv1d_transpose/strided_slice_1?
1conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:23
1conv1d_transpose/conv1d_transpose/concat/values_1?
-conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-conv1d_transpose/conv1d_transpose/concat/axis?
(conv1d_transpose/conv1d_transpose/concatConcatV28conv1d_transpose/conv1d_transpose/strided_slice:output:0:conv1d_transpose/conv1d_transpose/concat/values_1:output:0:conv1d_transpose/conv1d_transpose/strided_slice_1:output:06conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(conv1d_transpose/conv1d_transpose/concat?
!conv1d_transpose/conv1d_transposeConv2DBackpropInput1conv1d_transpose/conv1d_transpose/concat:output:07conv1d_transpose/conv1d_transpose/ExpandDims_1:output:05conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
2#
!conv1d_transpose/conv1d_transpose?
)conv1d_transpose/conv1d_transpose/SqueezeSqueeze*conv1d_transpose/conv1d_transpose:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2+
)conv1d_transpose/conv1d_transpose/Squeeze?
'conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv1d_transpose/BiasAdd/ReadVariableOp?
conv1d_transpose/BiasAddBiasAdd2conv1d_transpose/conv1d_transpose/Squeeze:output:0/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
conv1d_transpose/BiasAdd?
IdentityIdentity!conv1d_transpose/BiasAdd:output:0(^conv1d_transpose/BiasAdd/ReadVariableOp>^conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2R
'conv1d_transpose/BiasAdd/ReadVariableOp'conv1d_transpose/BiasAdd/ReadVariableOp2~
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__traced_save_160338
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop6
2savev2_conv1d_transpose_kernel_read_readvariableop4
0savev2_conv1d_transpose_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop2savev2_conv1d_transpose_kernel_read_readvariableop0savev2_conv1d_transpose_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*>
_input_shapes-
+: :	?:?:?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:)%
#
_output_shapes
:?: 

_output_shapes
::

_output_shapes
: 
?
?
$__inference_signature_wrapper_160125
input_1
unknown:	?
	unknown_0:	? 
	unknown_1:?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__wrapped_model_1599022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_160303

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:??????????2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
"__inference__traced_restore_160360
file_prefix0
assignvariableop_dense_kernel:	?,
assignvariableop_1_dense_bias:	?A
*assignvariableop_2_conv1d_transpose_kernel:?6
(assignvariableop_3_conv1d_transpose_bias:

identity_5??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp*assignvariableop_2_conv1d_transpose_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp(assignvariableop_3_conv1d_transpose_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????I
conv1d_transpose5
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
?&
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
._default_save_signature
/__call__
*0&call_and_return_all_conditional_losses"?$
_tf_keras_network?${"name": "CNN_decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "CNN_decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1732, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [433, 4]}}, "name": "reshape", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3056]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose", "inbound_nodes": [[["reshape", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv1d_transpose", 0, 0]]}, "shared_object_id": 8, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CNN_decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1732, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [433, 4]}}, "name": "reshape", "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3056]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose", "inbound_nodes": [[["reshape", 0, 0, {}]]], "shared_object_id": 7}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv1d_transpose", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
1__call__
*2&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1732, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
trainable_variables
	variables
regularization_losses
	keras_api
3__call__
*4&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [433, 4]}}, "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 4}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
5__call__
*6&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv1d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3056]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["reshape", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 433, 4]}}
<

0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<

0
1
2
3"
trackable_list_wrapper
?
non_trainable_variables
layer_metrics
layer_regularization_losses
trainable_variables
metrics
regularization_losses

layers
	variables
/__call__
._default_save_signature
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
,
7serving_default"
signature_map
:	?2dense/kernel
:?2
dense/bias
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
 layer_metrics
!layer_regularization_losses
trainable_variables
	variables
"metrics

#layers
regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
$non_trainable_variables
%layer_metrics
&layer_regularization_losses
trainable_variables
	variables
'metrics

(layers
regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
.:,?2conv1d_transpose/kernel
#:!2conv1d_transpose/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
)non_trainable_variables
*layer_metrics
+layer_regularization_losses
trainable_variables
	variables
,metrics

-layers
regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
!__inference__wrapped_model_159902?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
,__inference_CNN_decoder_layer_call_fn_160010
,__inference_CNN_decoder_layer_call_fn_160138
,__inference_CNN_decoder_layer_call_fn_160151
,__inference_CNN_decoder_layer_call_fn_160080?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_160208
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_160265
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_160095
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_160110?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_160274?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_160285?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_reshape_layer_call_fn_160290?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_reshape_layer_call_and_return_conditional_losses_160303?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_conv1d_transpose_layer_call_fn_159954?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_159944?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?B?
$__inference_signature_wrapper_160125input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_160095t
8?5
.?+
!?
input_1?????????
p 

 
? "2?/
(?%
0??????????????????
? ?
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_160110t
8?5
.?+
!?
input_1?????????
p

 
? "2?/
(?%
0??????????????????
? ?
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_160208k
7?4
-?*
 ?
inputs?????????
p 

 
? "*?'
 ?
0??????????
? ?
G__inference_CNN_decoder_layer_call_and_return_conditional_losses_160265k
7?4
-?*
 ?
inputs?????????
p

 
? "*?'
 ?
0??????????
? ?
,__inference_CNN_decoder_layer_call_fn_160010g
8?5
.?+
!?
input_1?????????
p 

 
? "%?"???????????????????
,__inference_CNN_decoder_layer_call_fn_160080g
8?5
.?+
!?
input_1?????????
p

 
? "%?"???????????????????
,__inference_CNN_decoder_layer_call_fn_160138f
7?4
-?*
 ?
inputs?????????
p 

 
? "%?"???????????????????
,__inference_CNN_decoder_layer_call_fn_160151f
7?4
-?*
 ?
inputs?????????
p

 
? "%?"???????????????????
!__inference__wrapped_model_159902?
0?-
&?#
!?
input_1?????????
? "H?E
C
conv1d_transpose/?,
conv1d_transpose???????????
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_159944v<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
1__inference_conv1d_transpose_layer_call_fn_159954i<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
A__inference_dense_layer_call_and_return_conditional_losses_160285]
/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? z
&__inference_dense_layer_call_fn_160274P
/?,
%?"
 ?
inputs?????????
? "????????????
C__inference_reshape_layer_call_and_return_conditional_losses_160303^0?-
&?#
!?
inputs??????????
? "*?'
 ?
0??????????
? }
(__inference_reshape_layer_call_fn_160290Q0?-
&?#
!?
inputs??????????
? "????????????
$__inference_signature_wrapper_160125?
;?8
? 
1?.
,
input_1!?
input_1?????????"H?E
C
conv1d_transpose/?,
conv1d_transpose??????????