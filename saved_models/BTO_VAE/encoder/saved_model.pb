??

??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
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
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
z
conv_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_0/kernel
s
!conv_0/kernel/Read/ReadVariableOpReadVariableOpconv_0/kernel*"
_output_shapes
:*
dtype0
n
conv_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_0/bias
g
conv_0/bias/Read/ReadVariableOpReadVariableOpconv_0/bias*
_output_shapes
:*
dtype0
z
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/kernel
s
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*"
_output_shapes
:*
dtype0
n
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/bias
g
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes
:*
dtype0
z
conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_2/kernel
s
!conv_2/kernel/Read/ReadVariableOpReadVariableOpconv_2/kernel*"
_output_shapes
:*
dtype0
n
conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_2/bias
g
conv_2/bias/Read/ReadVariableOpReadVariableOpconv_2/bias*
_output_shapes
:*
dtype0
w
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namez_mean/kernel
p
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes
:	?*
dtype0
n
z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_mean/bias
g
z_mean/bias/Read/ReadVariableOpReadVariableOpz_mean/bias*
_output_shapes
:*
dtype0
}
z_log_sig/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*!
shared_namez_log_sig/kernel
v
$z_log_sig/kernel/Read/ReadVariableOpReadVariableOpz_log_sig/kernel*
_output_shapes
:	?*
dtype0
t
z_log_sig/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_log_sig/bias
m
"z_log_sig/bias/Read/ReadVariableOpReadVariableOpz_log_sig/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?$
value?$B?$ B?$
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
R
!trainable_variables
"	variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'trainable_variables
(	variables
)regularization_losses
*	keras_api
R
+trainable_variables
,	variables
-regularization_losses
.	keras_api
R
/trainable_variables
0	variables
1regularization_losses
2	keras_api
R
3trainable_variables
4	variables
5regularization_losses
6	keras_api
h

7kernel
8bias
9trainable_variables
:	variables
;regularization_losses
<	keras_api
h

=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
F
0
1
2
3
%4
&5
76
87
=8
>9
 
F
0
1
2
3
%4
&5
76
87
=8
>9
?
Cnon_trainable_variables
Dlayer_metrics
Elayer_regularization_losses
trainable_variables
Fmetrics
regularization_losses

Glayers
	variables
 
YW
VARIABLE_VALUEconv_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Hnon_trainable_variables
Ilayer_metrics
Jlayer_regularization_losses
trainable_variables
	variables
Kmetrics

Llayers
regularization_losses
 
 
 
?
Mnon_trainable_variables
Nlayer_metrics
Olayer_regularization_losses
trainable_variables
	variables
Pmetrics

Qlayers
regularization_losses
YW
VARIABLE_VALUEconv_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Rnon_trainable_variables
Slayer_metrics
Tlayer_regularization_losses
trainable_variables
	variables
Umetrics

Vlayers
regularization_losses
 
 
 
?
Wnon_trainable_variables
Xlayer_metrics
Ylayer_regularization_losses
!trainable_variables
"	variables
Zmetrics

[layers
#regularization_losses
YW
VARIABLE_VALUEconv_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
?
\non_trainable_variables
]layer_metrics
^layer_regularization_losses
'trainable_variables
(	variables
_metrics

`layers
)regularization_losses
 
 
 
?
anon_trainable_variables
blayer_metrics
clayer_regularization_losses
+trainable_variables
,	variables
dmetrics

elayers
-regularization_losses
 
 
 
?
fnon_trainable_variables
glayer_metrics
hlayer_regularization_losses
/trainable_variables
0	variables
imetrics

jlayers
1regularization_losses
 
 
 
?
knon_trainable_variables
llayer_metrics
mlayer_regularization_losses
3trainable_variables
4	variables
nmetrics

olayers
5regularization_losses
YW
VARIABLE_VALUEz_mean/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEz_mean/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81

70
81
 
?
pnon_trainable_variables
qlayer_metrics
rlayer_regularization_losses
9trainable_variables
:	variables
smetrics

tlayers
;regularization_losses
\Z
VARIABLE_VALUEz_log_sig/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEz_log_sig/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
 
?
unon_trainable_variables
vlayer_metrics
wlayer_regularization_losses
?trainable_variables
@	variables
xmetrics

ylayers
Aregularization_losses
 
 
 
 
N
0
1
2
3
4
5
6
7
	8

9
10
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
 
 
 
 
 
~
serving_default_XPlaceholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_Xconv_0/kernelconv_0/biasconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasz_log_sig/kernelz_log_sig/biasz_mean/kernelz_mean/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8? *-
f(R&
$__inference_signature_wrapper_159372
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_0/kernel/Read/ReadVariableOpconv_0/bias/Read/ReadVariableOp!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOp$z_log_sig/kernel/Read/ReadVariableOp"z_log_sig/bias/Read/ReadVariableOpConst*
Tin
2*
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
__inference__traced_save_159749
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_0/kernelconv_0/biasconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasz_mean/kernelz_mean/biasz_log_sig/kernelz_log_sig/bias*
Tin
2*
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
"__inference__traced_restore_159789??
?[
?
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159555

inputsH
2conv_0_conv1d_expanddims_1_readvariableop_resource:4
&conv_0_biasadd_readvariableop_resource:H
2conv_1_conv1d_expanddims_1_readvariableop_resource:4
&conv_1_biasadd_readvariableop_resource:H
2conv_2_conv1d_expanddims_1_readvariableop_resource:4
&conv_2_biasadd_readvariableop_resource:;
(z_log_sig_matmul_readvariableop_resource:	?7
)z_log_sig_biasadd_readvariableop_resource:8
%z_mean_matmul_readvariableop_resource:	?4
&z_mean_biasadd_readvariableop_resource:
identity

identity_1??conv_0/BiasAdd/ReadVariableOp?)conv_0/conv1d/ExpandDims_1/ReadVariableOp?conv_1/BiasAdd/ReadVariableOp?)conv_1/conv1d/ExpandDims_1/ReadVariableOp?conv_2/BiasAdd/ReadVariableOp?)conv_2/conv1d/ExpandDims_1/ReadVariableOp? z_log_sig/BiasAdd/ReadVariableOp?z_log_sig/MatMul/ReadVariableOp?z_mean/BiasAdd/ReadVariableOp?z_mean/MatMul/ReadVariableOp?
conv_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv_0/conv1d/ExpandDims/dim?
conv_0/conv1d/ExpandDims
ExpandDimsinputs%conv_0/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv_0/conv1d/ExpandDims?
)conv_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv_0/conv1d/ExpandDims_1/ReadVariableOp?
conv_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_0/conv1d/ExpandDims_1/dim?
conv_0/conv1d/ExpandDims_1
ExpandDims1conv_0/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_0/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv_0/conv1d/ExpandDims_1?
conv_0/conv1dConv2D!conv_0/conv1d/ExpandDims:output:0#conv_0/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv_0/conv1d?
conv_0/conv1d/SqueezeSqueezeconv_0/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv_0/conv1d/Squeeze?
conv_0/BiasAdd/ReadVariableOpReadVariableOp&conv_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_0/BiasAdd/ReadVariableOp?
conv_0/BiasAddBiasAddconv_0/conv1d/Squeeze:output:0%conv_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
conv_0/BiasAdd?
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dim?
average_pooling1d/ExpandDims
ExpandDimsconv_0/BiasAdd:output:0)average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
average_pooling1d/ExpandDims?
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2
average_pooling1d/AvgPool?
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2
average_pooling1d/Squeeze?
conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv_1/conv1d/ExpandDims/dim?
conv_1/conv1d/ExpandDims
ExpandDims"average_pooling1d/Squeeze:output:0%conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv_1/conv1d/ExpandDims?
)conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv_1/conv1d/ExpandDims_1/ReadVariableOp?
conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_1/conv1d/ExpandDims_1/dim?
conv_1/conv1d/ExpandDims_1
ExpandDims1conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv_1/conv1d/ExpandDims_1?
conv_1/conv1dConv2D!conv_1/conv1d/ExpandDims:output:0#conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv_1/conv1d?
conv_1/conv1d/SqueezeSqueezeconv_1/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv_1/conv1d/Squeeze?
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_1/BiasAdd/ReadVariableOp?
conv_1/BiasAddBiasAddconv_1/conv1d/Squeeze:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
conv_1/BiasAdd?
"average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_1/ExpandDims/dim?
average_pooling1d_1/ExpandDims
ExpandDimsconv_1/BiasAdd:output:0+average_pooling1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2 
average_pooling1d_1/ExpandDims?
average_pooling1d_1/AvgPoolAvgPool'average_pooling1d_1/ExpandDims:output:0*
T0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2
average_pooling1d_1/AvgPool?
average_pooling1d_1/SqueezeSqueeze$average_pooling1d_1/AvgPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2
average_pooling1d_1/Squeeze?
conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv_2/conv1d/ExpandDims/dim?
conv_2/conv1d/ExpandDims
ExpandDims$average_pooling1d_1/Squeeze:output:0%conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv_2/conv1d/ExpandDims?
)conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv_2/conv1d/ExpandDims_1/ReadVariableOp?
conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_2/conv1d/ExpandDims_1/dim?
conv_2/conv1d/ExpandDims_1
ExpandDims1conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv_2/conv1d/ExpandDims_1?
conv_2/conv1dConv2D!conv_2/conv1d/ExpandDims:output:0#conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv_2/conv1d?
conv_2/conv1d/SqueezeSqueezeconv_2/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv_2/conv1d/Squeeze?
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_2/BiasAdd/ReadVariableOp?
conv_2/BiasAddBiasAddconv_2/conv1d/Squeeze:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
conv_2/BiasAdd?
"average_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_2/ExpandDims/dim?
average_pooling1d_2/ExpandDims
ExpandDimsconv_2/BiasAdd:output:0+average_pooling1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2 
average_pooling1d_2/ExpandDims?
average_pooling1d_2/AvgPoolAvgPool'average_pooling1d_2/ExpandDims:output:0*
T0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2
average_pooling1d_2/AvgPool?
average_pooling1d_2/SqueezeSqueeze$average_pooling1d_2/AvgPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2
average_pooling1d_2/Squeezeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshape$average_pooling1d_2/Squeeze:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
z_log_sig/MatMul/ReadVariableOpReadVariableOp(z_log_sig_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
z_log_sig/MatMul/ReadVariableOp?
z_log_sig/MatMulMatMulflatten/Reshape:output:0'z_log_sig/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_sig/MatMul?
 z_log_sig/BiasAdd/ReadVariableOpReadVariableOp)z_log_sig_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_sig/BiasAdd/ReadVariableOp?
z_log_sig/BiasAddBiasAddz_log_sig/MatMul:product:0(z_log_sig/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_sig/BiasAdd?
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
z_mean/MatMul/ReadVariableOp?
z_mean/MatMulMatMulflatten/Reshape:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/MatMul?
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOp?
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/BiasAdd?
IdentityIdentityz_mean/BiasAdd:output:0^conv_0/BiasAdd/ReadVariableOp*^conv_0/conv1d/ExpandDims_1/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp*^conv_1/conv1d/ExpandDims_1/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp*^conv_2/conv1d/ExpandDims_1/ReadVariableOp!^z_log_sig/BiasAdd/ReadVariableOp ^z_log_sig/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identityz_log_sig/BiasAdd:output:0^conv_0/BiasAdd/ReadVariableOp*^conv_0/conv1d/ExpandDims_1/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp*^conv_1/conv1d/ExpandDims_1/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp*^conv_2/conv1d/ExpandDims_1/ReadVariableOp!^z_log_sig/BiasAdd/ReadVariableOp ^z_log_sig/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????: : : : : : : : : : 2>
conv_0/BiasAdd/ReadVariableOpconv_0/BiasAdd/ReadVariableOp2V
)conv_0/conv1d/ExpandDims_1/ReadVariableOp)conv_0/conv1d/ExpandDims_1/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2V
)conv_1/conv1d/ExpandDims_1/ReadVariableOp)conv_1/conv1d/ExpandDims_1/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2V
)conv_2/conv1d/ExpandDims_1/ReadVariableOp)conv_2/conv1d/ExpandDims_1/ReadVariableOp2D
 z_log_sig/BiasAdd/ReadVariableOp z_log_sig/BiasAdd/ReadVariableOp2B
z_log_sig/MatMul/ReadVariableOpz_log_sig/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_dropout_layer_call_and_return_conditional_losses_159118

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
N
2__inference_average_pooling1d_layer_call_fn_158909

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_1589032
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_layer_call_fn_159643

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1590252
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
i
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_158903

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
2	
AvgPool?
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_layer_call_fn_159648

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1591182
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_conv_1_layer_call_and_return_conditional_losses_159603

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159491

inputsH
2conv_0_conv1d_expanddims_1_readvariableop_resource:4
&conv_0_biasadd_readvariableop_resource:H
2conv_1_conv1d_expanddims_1_readvariableop_resource:4
&conv_1_biasadd_readvariableop_resource:H
2conv_2_conv1d_expanddims_1_readvariableop_resource:4
&conv_2_biasadd_readvariableop_resource:;
(z_log_sig_matmul_readvariableop_resource:	?7
)z_log_sig_biasadd_readvariableop_resource:8
%z_mean_matmul_readvariableop_resource:	?4
&z_mean_biasadd_readvariableop_resource:
identity

identity_1??conv_0/BiasAdd/ReadVariableOp?)conv_0/conv1d/ExpandDims_1/ReadVariableOp?conv_1/BiasAdd/ReadVariableOp?)conv_1/conv1d/ExpandDims_1/ReadVariableOp?conv_2/BiasAdd/ReadVariableOp?)conv_2/conv1d/ExpandDims_1/ReadVariableOp? z_log_sig/BiasAdd/ReadVariableOp?z_log_sig/MatMul/ReadVariableOp?z_mean/BiasAdd/ReadVariableOp?z_mean/MatMul/ReadVariableOp?
conv_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv_0/conv1d/ExpandDims/dim?
conv_0/conv1d/ExpandDims
ExpandDimsinputs%conv_0/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv_0/conv1d/ExpandDims?
)conv_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv_0/conv1d/ExpandDims_1/ReadVariableOp?
conv_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_0/conv1d/ExpandDims_1/dim?
conv_0/conv1d/ExpandDims_1
ExpandDims1conv_0/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_0/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv_0/conv1d/ExpandDims_1?
conv_0/conv1dConv2D!conv_0/conv1d/ExpandDims:output:0#conv_0/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv_0/conv1d?
conv_0/conv1d/SqueezeSqueezeconv_0/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv_0/conv1d/Squeeze?
conv_0/BiasAdd/ReadVariableOpReadVariableOp&conv_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_0/BiasAdd/ReadVariableOp?
conv_0/BiasAddBiasAddconv_0/conv1d/Squeeze:output:0%conv_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
conv_0/BiasAdd?
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dim?
average_pooling1d/ExpandDims
ExpandDimsconv_0/BiasAdd:output:0)average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
average_pooling1d/ExpandDims?
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2
average_pooling1d/AvgPool?
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2
average_pooling1d/Squeeze?
conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv_1/conv1d/ExpandDims/dim?
conv_1/conv1d/ExpandDims
ExpandDims"average_pooling1d/Squeeze:output:0%conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv_1/conv1d/ExpandDims?
)conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv_1/conv1d/ExpandDims_1/ReadVariableOp?
conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_1/conv1d/ExpandDims_1/dim?
conv_1/conv1d/ExpandDims_1
ExpandDims1conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv_1/conv1d/ExpandDims_1?
conv_1/conv1dConv2D!conv_1/conv1d/ExpandDims:output:0#conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv_1/conv1d?
conv_1/conv1d/SqueezeSqueezeconv_1/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv_1/conv1d/Squeeze?
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_1/BiasAdd/ReadVariableOp?
conv_1/BiasAddBiasAddconv_1/conv1d/Squeeze:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
conv_1/BiasAdd?
"average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_1/ExpandDims/dim?
average_pooling1d_1/ExpandDims
ExpandDimsconv_1/BiasAdd:output:0+average_pooling1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2 
average_pooling1d_1/ExpandDims?
average_pooling1d_1/AvgPoolAvgPool'average_pooling1d_1/ExpandDims:output:0*
T0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2
average_pooling1d_1/AvgPool?
average_pooling1d_1/SqueezeSqueeze$average_pooling1d_1/AvgPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2
average_pooling1d_1/Squeeze?
conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv_2/conv1d/ExpandDims/dim?
conv_2/conv1d/ExpandDims
ExpandDims$average_pooling1d_1/Squeeze:output:0%conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv_2/conv1d/ExpandDims?
)conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv_2/conv1d/ExpandDims_1/ReadVariableOp?
conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_2/conv1d/ExpandDims_1/dim?
conv_2/conv1d/ExpandDims_1
ExpandDims1conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv_2/conv1d/ExpandDims_1?
conv_2/conv1dConv2D!conv_2/conv1d/ExpandDims:output:0#conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv_2/conv1d?
conv_2/conv1d/SqueezeSqueezeconv_2/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv_2/conv1d/Squeeze?
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_2/BiasAdd/ReadVariableOp?
conv_2/BiasAddBiasAddconv_2/conv1d/Squeeze:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
conv_2/BiasAdd?
"average_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_2/ExpandDims/dim?
average_pooling1d_2/ExpandDims
ExpandDimsconv_2/BiasAdd:output:0+average_pooling1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2 
average_pooling1d_2/ExpandDims?
average_pooling1d_2/AvgPoolAvgPool'average_pooling1d_2/ExpandDims:output:0*
T0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2
average_pooling1d_2/AvgPool?
average_pooling1d_2/SqueezeSqueeze$average_pooling1d_2/AvgPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2
average_pooling1d_2/Squeezeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshape$average_pooling1d_2/Squeeze:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape}
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
z_log_sig/MatMul/ReadVariableOpReadVariableOp(z_log_sig_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
z_log_sig/MatMul/ReadVariableOp?
z_log_sig/MatMulMatMuldropout/Identity:output:0'z_log_sig/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_sig/MatMul?
 z_log_sig/BiasAdd/ReadVariableOpReadVariableOp)z_log_sig_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_sig/BiasAdd/ReadVariableOp?
z_log_sig/BiasAddBiasAddz_log_sig/MatMul:product:0(z_log_sig/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_sig/BiasAdd?
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
z_mean/MatMul/ReadVariableOp?
z_mean/MatMulMatMuldropout/Identity:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/MatMul?
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOp?
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/BiasAdd?
IdentityIdentityz_mean/BiasAdd:output:0^conv_0/BiasAdd/ReadVariableOp*^conv_0/conv1d/ExpandDims_1/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp*^conv_1/conv1d/ExpandDims_1/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp*^conv_2/conv1d/ExpandDims_1/ReadVariableOp!^z_log_sig/BiasAdd/ReadVariableOp ^z_log_sig/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identityz_log_sig/BiasAdd:output:0^conv_0/BiasAdd/ReadVariableOp*^conv_0/conv1d/ExpandDims_1/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp*^conv_1/conv1d/ExpandDims_1/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp*^conv_2/conv1d/ExpandDims_1/ReadVariableOp!^z_log_sig/BiasAdd/ReadVariableOp ^z_log_sig/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????: : : : : : : : : : 2>
conv_0/BiasAdd/ReadVariableOpconv_0/BiasAdd/ReadVariableOp2V
)conv_0/conv1d/ExpandDims_1/ReadVariableOp)conv_0/conv1d/ExpandDims_1/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2V
)conv_1/conv1d/ExpandDims_1/ReadVariableOp)conv_1/conv1d/ExpandDims_1/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2V
)conv_2/conv1d/ExpandDims_1/ReadVariableOp)conv_2/conv1d/ExpandDims_1/ReadVariableOp2D
 z_log_sig/BiasAdd/ReadVariableOp z_log_sig/BiasAdd/ReadVariableOp2B
z_log_sig/MatMul/ReadVariableOpz_log_sig/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_conv_1_layer_call_and_return_conditional_losses_158983

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_159018

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
"__inference__traced_restore_159789
file_prefix4
assignvariableop_conv_0_kernel:,
assignvariableop_1_conv_0_bias:6
 assignvariableop_2_conv_1_kernel:,
assignvariableop_3_conv_1_bias:6
 assignvariableop_4_conv_2_kernel:,
assignvariableop_5_conv_2_bias:3
 assignvariableop_6_z_mean_kernel:	?,
assignvariableop_7_z_mean_bias:6
#assignvariableop_8_z_log_sig_kernel:	?/
!assignvariableop_9_z_log_sig_bias:
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_z_mean_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_z_mean_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_z_log_sig_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_z_log_sig_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10?
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
P
4__inference_average_pooling1d_1_layer_call_fn_158924

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_1589182
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_conv_1_layer_call_fn_159588

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_1589832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159221

inputs#
conv_0_159189:
conv_0_159191:#
conv_1_159195:
conv_1_159197:#
conv_2_159201:
conv_2_159203:#
z_log_sig_159209:	?
z_log_sig_159211: 
z_mean_159214:	?
z_mean_159216:
identity

identity_1??conv_0/StatefulPartitionedCall?conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?!z_log_sig/StatefulPartitionedCall?z_mean/StatefulPartitionedCall?
conv_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv_0_159189conv_0_159191*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_0_layer_call_and_return_conditional_losses_1589612 
conv_0/StatefulPartitionedCall?
!average_pooling1d/PartitionedCallPartitionedCall'conv_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_1589032#
!average_pooling1d/PartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0conv_1_159195conv_1_159197*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_1589832 
conv_1/StatefulPartitionedCall?
#average_pooling1d_1/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_1589182%
#average_pooling1d_1/PartitionedCall?
conv_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0conv_2_159201conv_2_159203*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_1590052 
conv_2/StatefulPartitionedCall?
#average_pooling1d_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *X
fSRQ
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_1589332%
#average_pooling1d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall,average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1590182
flatten/PartitionedCall?
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1591182
dropout/PartitionedCall?
!z_log_sig/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0z_log_sig_159209z_log_sig_159211*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_z_log_sig_layer_call_and_return_conditional_losses_1590372#
!z_log_sig/StatefulPartitionedCall?
z_mean/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0z_mean_159214z_mean_159216*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1590532 
z_mean/StatefulPartitionedCall?
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall"^z_log_sig/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*z_log_sig/StatefulPartitionedCall:output:0^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall"^z_log_sig/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????: : : : : : : : : : 2@
conv_0/StatefulPartitionedCallconv_0/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2F
!z_log_sig/StatefulPartitionedCall!z_log_sig/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
__inference__traced_save_159749
file_prefix,
(savev2_conv_0_kernel_read_readvariableop*
&savev2_conv_0_bias_read_readvariableop,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop,
(savev2_z_mean_kernel_read_readvariableop*
&savev2_z_mean_bias_read_readvariableop/
+savev2_z_log_sig_kernel_read_readvariableop-
)savev2_z_log_sig_bias_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_0_kernel_read_readvariableop&savev2_conv_0_bias_read_readvariableop(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_sig_kernel_read_readvariableop)savev2_z_log_sig_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*u
_input_shapesd
b: :::::::	?::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%	!

_output_shapes
:	?: 


_output_shapes
::

_output_shapes
: 
?q
?

!__inference__wrapped_model_158894
xT
>cnn_encoder_conv_0_conv1d_expanddims_1_readvariableop_resource:@
2cnn_encoder_conv_0_biasadd_readvariableop_resource:T
>cnn_encoder_conv_1_conv1d_expanddims_1_readvariableop_resource:@
2cnn_encoder_conv_1_biasadd_readvariableop_resource:T
>cnn_encoder_conv_2_conv1d_expanddims_1_readvariableop_resource:@
2cnn_encoder_conv_2_biasadd_readvariableop_resource:G
4cnn_encoder_z_log_sig_matmul_readvariableop_resource:	?C
5cnn_encoder_z_log_sig_biasadd_readvariableop_resource:D
1cnn_encoder_z_mean_matmul_readvariableop_resource:	?@
2cnn_encoder_z_mean_biasadd_readvariableop_resource:
identity

identity_1??)CNN_encoder/conv_0/BiasAdd/ReadVariableOp?5CNN_encoder/conv_0/conv1d/ExpandDims_1/ReadVariableOp?)CNN_encoder/conv_1/BiasAdd/ReadVariableOp?5CNN_encoder/conv_1/conv1d/ExpandDims_1/ReadVariableOp?)CNN_encoder/conv_2/BiasAdd/ReadVariableOp?5CNN_encoder/conv_2/conv1d/ExpandDims_1/ReadVariableOp?,CNN_encoder/z_log_sig/BiasAdd/ReadVariableOp?+CNN_encoder/z_log_sig/MatMul/ReadVariableOp?)CNN_encoder/z_mean/BiasAdd/ReadVariableOp?(CNN_encoder/z_mean/MatMul/ReadVariableOp?
(CNN_encoder/conv_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(CNN_encoder/conv_0/conv1d/ExpandDims/dim?
$CNN_encoder/conv_0/conv1d/ExpandDims
ExpandDimsx1CNN_encoder/conv_0/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2&
$CNN_encoder/conv_0/conv1d/ExpandDims?
5CNN_encoder/conv_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>cnn_encoder_conv_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype027
5CNN_encoder/conv_0/conv1d/ExpandDims_1/ReadVariableOp?
*CNN_encoder/conv_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*CNN_encoder/conv_0/conv1d/ExpandDims_1/dim?
&CNN_encoder/conv_0/conv1d/ExpandDims_1
ExpandDims=CNN_encoder/conv_0/conv1d/ExpandDims_1/ReadVariableOp:value:03CNN_encoder/conv_0/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2(
&CNN_encoder/conv_0/conv1d/ExpandDims_1?
CNN_encoder/conv_0/conv1dConv2D-CNN_encoder/conv_0/conv1d/ExpandDims:output:0/CNN_encoder/conv_0/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
CNN_encoder/conv_0/conv1d?
!CNN_encoder/conv_0/conv1d/SqueezeSqueeze"CNN_encoder/conv_0/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2#
!CNN_encoder/conv_0/conv1d/Squeeze?
)CNN_encoder/conv_0/BiasAdd/ReadVariableOpReadVariableOp2cnn_encoder_conv_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)CNN_encoder/conv_0/BiasAdd/ReadVariableOp?
CNN_encoder/conv_0/BiasAddBiasAdd*CNN_encoder/conv_0/conv1d/Squeeze:output:01CNN_encoder/conv_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
CNN_encoder/conv_0/BiasAdd?
,CNN_encoder/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,CNN_encoder/average_pooling1d/ExpandDims/dim?
(CNN_encoder/average_pooling1d/ExpandDims
ExpandDims#CNN_encoder/conv_0/BiasAdd:output:05CNN_encoder/average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2*
(CNN_encoder/average_pooling1d/ExpandDims?
%CNN_encoder/average_pooling1d/AvgPoolAvgPool1CNN_encoder/average_pooling1d/ExpandDims:output:0*
T0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2'
%CNN_encoder/average_pooling1d/AvgPool?
%CNN_encoder/average_pooling1d/SqueezeSqueeze.CNN_encoder/average_pooling1d/AvgPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2'
%CNN_encoder/average_pooling1d/Squeeze?
(CNN_encoder/conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(CNN_encoder/conv_1/conv1d/ExpandDims/dim?
$CNN_encoder/conv_1/conv1d/ExpandDims
ExpandDims.CNN_encoder/average_pooling1d/Squeeze:output:01CNN_encoder/conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2&
$CNN_encoder/conv_1/conv1d/ExpandDims?
5CNN_encoder/conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>cnn_encoder_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype027
5CNN_encoder/conv_1/conv1d/ExpandDims_1/ReadVariableOp?
*CNN_encoder/conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*CNN_encoder/conv_1/conv1d/ExpandDims_1/dim?
&CNN_encoder/conv_1/conv1d/ExpandDims_1
ExpandDims=CNN_encoder/conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:03CNN_encoder/conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2(
&CNN_encoder/conv_1/conv1d/ExpandDims_1?
CNN_encoder/conv_1/conv1dConv2D-CNN_encoder/conv_1/conv1d/ExpandDims:output:0/CNN_encoder/conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
CNN_encoder/conv_1/conv1d?
!CNN_encoder/conv_1/conv1d/SqueezeSqueeze"CNN_encoder/conv_1/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2#
!CNN_encoder/conv_1/conv1d/Squeeze?
)CNN_encoder/conv_1/BiasAdd/ReadVariableOpReadVariableOp2cnn_encoder_conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)CNN_encoder/conv_1/BiasAdd/ReadVariableOp?
CNN_encoder/conv_1/BiasAddBiasAdd*CNN_encoder/conv_1/conv1d/Squeeze:output:01CNN_encoder/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
CNN_encoder/conv_1/BiasAdd?
.CNN_encoder/average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :20
.CNN_encoder/average_pooling1d_1/ExpandDims/dim?
*CNN_encoder/average_pooling1d_1/ExpandDims
ExpandDims#CNN_encoder/conv_1/BiasAdd:output:07CNN_encoder/average_pooling1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2,
*CNN_encoder/average_pooling1d_1/ExpandDims?
'CNN_encoder/average_pooling1d_1/AvgPoolAvgPool3CNN_encoder/average_pooling1d_1/ExpandDims:output:0*
T0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2)
'CNN_encoder/average_pooling1d_1/AvgPool?
'CNN_encoder/average_pooling1d_1/SqueezeSqueeze0CNN_encoder/average_pooling1d_1/AvgPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2)
'CNN_encoder/average_pooling1d_1/Squeeze?
(CNN_encoder/conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(CNN_encoder/conv_2/conv1d/ExpandDims/dim?
$CNN_encoder/conv_2/conv1d/ExpandDims
ExpandDims0CNN_encoder/average_pooling1d_1/Squeeze:output:01CNN_encoder/conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2&
$CNN_encoder/conv_2/conv1d/ExpandDims?
5CNN_encoder/conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>cnn_encoder_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype027
5CNN_encoder/conv_2/conv1d/ExpandDims_1/ReadVariableOp?
*CNN_encoder/conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*CNN_encoder/conv_2/conv1d/ExpandDims_1/dim?
&CNN_encoder/conv_2/conv1d/ExpandDims_1
ExpandDims=CNN_encoder/conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:03CNN_encoder/conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2(
&CNN_encoder/conv_2/conv1d/ExpandDims_1?
CNN_encoder/conv_2/conv1dConv2D-CNN_encoder/conv_2/conv1d/ExpandDims:output:0/CNN_encoder/conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
CNN_encoder/conv_2/conv1d?
!CNN_encoder/conv_2/conv1d/SqueezeSqueeze"CNN_encoder/conv_2/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2#
!CNN_encoder/conv_2/conv1d/Squeeze?
)CNN_encoder/conv_2/BiasAdd/ReadVariableOpReadVariableOp2cnn_encoder_conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)CNN_encoder/conv_2/BiasAdd/ReadVariableOp?
CNN_encoder/conv_2/BiasAddBiasAdd*CNN_encoder/conv_2/conv1d/Squeeze:output:01CNN_encoder/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
CNN_encoder/conv_2/BiasAdd?
.CNN_encoder/average_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :20
.CNN_encoder/average_pooling1d_2/ExpandDims/dim?
*CNN_encoder/average_pooling1d_2/ExpandDims
ExpandDims#CNN_encoder/conv_2/BiasAdd:output:07CNN_encoder/average_pooling1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2,
*CNN_encoder/average_pooling1d_2/ExpandDims?
'CNN_encoder/average_pooling1d_2/AvgPoolAvgPool3CNN_encoder/average_pooling1d_2/ExpandDims:output:0*
T0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2)
'CNN_encoder/average_pooling1d_2/AvgPool?
'CNN_encoder/average_pooling1d_2/SqueezeSqueeze0CNN_encoder/average_pooling1d_2/AvgPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2)
'CNN_encoder/average_pooling1d_2/Squeeze?
CNN_encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
CNN_encoder/flatten/Const?
CNN_encoder/flatten/ReshapeReshape0CNN_encoder/average_pooling1d_2/Squeeze:output:0"CNN_encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
CNN_encoder/flatten/Reshape?
CNN_encoder/dropout/IdentityIdentity$CNN_encoder/flatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2
CNN_encoder/dropout/Identity?
+CNN_encoder/z_log_sig/MatMul/ReadVariableOpReadVariableOp4cnn_encoder_z_log_sig_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02-
+CNN_encoder/z_log_sig/MatMul/ReadVariableOp?
CNN_encoder/z_log_sig/MatMulMatMul%CNN_encoder/dropout/Identity:output:03CNN_encoder/z_log_sig/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
CNN_encoder/z_log_sig/MatMul?
,CNN_encoder/z_log_sig/BiasAdd/ReadVariableOpReadVariableOp5cnn_encoder_z_log_sig_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,CNN_encoder/z_log_sig/BiasAdd/ReadVariableOp?
CNN_encoder/z_log_sig/BiasAddBiasAdd&CNN_encoder/z_log_sig/MatMul:product:04CNN_encoder/z_log_sig/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
CNN_encoder/z_log_sig/BiasAdd?
(CNN_encoder/z_mean/MatMul/ReadVariableOpReadVariableOp1cnn_encoder_z_mean_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(CNN_encoder/z_mean/MatMul/ReadVariableOp?
CNN_encoder/z_mean/MatMulMatMul%CNN_encoder/dropout/Identity:output:00CNN_encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
CNN_encoder/z_mean/MatMul?
)CNN_encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp2cnn_encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)CNN_encoder/z_mean/BiasAdd/ReadVariableOp?
CNN_encoder/z_mean/BiasAddBiasAdd#CNN_encoder/z_mean/MatMul:product:01CNN_encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
CNN_encoder/z_mean/BiasAdd?
IdentityIdentity&CNN_encoder/z_log_sig/BiasAdd:output:0*^CNN_encoder/conv_0/BiasAdd/ReadVariableOp6^CNN_encoder/conv_0/conv1d/ExpandDims_1/ReadVariableOp*^CNN_encoder/conv_1/BiasAdd/ReadVariableOp6^CNN_encoder/conv_1/conv1d/ExpandDims_1/ReadVariableOp*^CNN_encoder/conv_2/BiasAdd/ReadVariableOp6^CNN_encoder/conv_2/conv1d/ExpandDims_1/ReadVariableOp-^CNN_encoder/z_log_sig/BiasAdd/ReadVariableOp,^CNN_encoder/z_log_sig/MatMul/ReadVariableOp*^CNN_encoder/z_mean/BiasAdd/ReadVariableOp)^CNN_encoder/z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity#CNN_encoder/z_mean/BiasAdd:output:0*^CNN_encoder/conv_0/BiasAdd/ReadVariableOp6^CNN_encoder/conv_0/conv1d/ExpandDims_1/ReadVariableOp*^CNN_encoder/conv_1/BiasAdd/ReadVariableOp6^CNN_encoder/conv_1/conv1d/ExpandDims_1/ReadVariableOp*^CNN_encoder/conv_2/BiasAdd/ReadVariableOp6^CNN_encoder/conv_2/conv1d/ExpandDims_1/ReadVariableOp-^CNN_encoder/z_log_sig/BiasAdd/ReadVariableOp,^CNN_encoder/z_log_sig/MatMul/ReadVariableOp*^CNN_encoder/z_mean/BiasAdd/ReadVariableOp)^CNN_encoder/z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????: : : : : : : : : : 2V
)CNN_encoder/conv_0/BiasAdd/ReadVariableOp)CNN_encoder/conv_0/BiasAdd/ReadVariableOp2n
5CNN_encoder/conv_0/conv1d/ExpandDims_1/ReadVariableOp5CNN_encoder/conv_0/conv1d/ExpandDims_1/ReadVariableOp2V
)CNN_encoder/conv_1/BiasAdd/ReadVariableOp)CNN_encoder/conv_1/BiasAdd/ReadVariableOp2n
5CNN_encoder/conv_1/conv1d/ExpandDims_1/ReadVariableOp5CNN_encoder/conv_1/conv1d/ExpandDims_1/ReadVariableOp2V
)CNN_encoder/conv_2/BiasAdd/ReadVariableOp)CNN_encoder/conv_2/BiasAdd/ReadVariableOp2n
5CNN_encoder/conv_2/conv1d/ExpandDims_1/ReadVariableOp5CNN_encoder/conv_2/conv1d/ExpandDims_1/ReadVariableOp2\
,CNN_encoder/z_log_sig/BiasAdd/ReadVariableOp,CNN_encoder/z_log_sig/BiasAdd/ReadVariableOp2Z
+CNN_encoder/z_log_sig/MatMul/ReadVariableOp+CNN_encoder/z_log_sig/MatMul/ReadVariableOp2V
)CNN_encoder/z_mean/BiasAdd/ReadVariableOp)CNN_encoder/z_mean/BiasAdd/ReadVariableOp2T
(CNN_encoder/z_mean/MatMul/ReadVariableOp(CNN_encoder/z_mean/MatMul/ReadVariableOp:O K
,
_output_shapes
:??????????

_user_specified_nameX
?.
?
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159061

inputs#
conv_0_158962:
conv_0_158964:#
conv_1_158984:
conv_1_158986:#
conv_2_159006:
conv_2_159008:#
z_log_sig_159038:	?
z_log_sig_159040: 
z_mean_159054:	?
z_mean_159056:
identity

identity_1??conv_0/StatefulPartitionedCall?conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?!z_log_sig/StatefulPartitionedCall?z_mean/StatefulPartitionedCall?
conv_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv_0_158962conv_0_158964*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_0_layer_call_and_return_conditional_losses_1589612 
conv_0/StatefulPartitionedCall?
!average_pooling1d/PartitionedCallPartitionedCall'conv_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_1589032#
!average_pooling1d/PartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0conv_1_158984conv_1_158986*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_1589832 
conv_1/StatefulPartitionedCall?
#average_pooling1d_1/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_1589182%
#average_pooling1d_1/PartitionedCall?
conv_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0conv_2_159006conv_2_159008*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_1590052 
conv_2/StatefulPartitionedCall?
#average_pooling1d_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *X
fSRQ
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_1589332%
#average_pooling1d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall,average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1590182
flatten/PartitionedCall?
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1590252
dropout/PartitionedCall?
!z_log_sig/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0z_log_sig_159038z_log_sig_159040*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_z_log_sig_layer_call_and_return_conditional_losses_1590372#
!z_log_sig/StatefulPartitionedCall?
z_mean/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0z_mean_159054z_mean_159056*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1590532 
z_mean/StatefulPartitionedCall?
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall"^z_log_sig/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*z_log_sig/StatefulPartitionedCall:output:0^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall"^z_log_sig/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????: : : : : : : : : : 2@
conv_0/StatefulPartitionedCallconv_0/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2F
!z_log_sig/StatefulPartitionedCall!z_log_sig/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_159653

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_z_log_sig_layer_call_and_return_conditional_losses_159037

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_159372
x
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7:	?
	unknown_8:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__wrapped_model_1588942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????

_user_specified_nameX
?
?
,__inference_CNN_encoder_layer_call_fn_159273
x
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7:	?
	unknown_8:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_1592212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????

_user_specified_nameX
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_159025

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_conv_2_layer_call_and_return_conditional_losses_159005

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
B__inference_z_mean_layer_call_and_return_conditional_losses_159053

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_CNN_encoder_layer_call_fn_159086
x
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7:	?
	unknown_8:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_1590612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????

_user_specified_nameX
?
P
4__inference_average_pooling1d_2_layer_call_fn_158939

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *X
fSRQ
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_1589332
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_CNN_encoder_layer_call_fn_159399

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7:	?
	unknown_8:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_1590612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159343
x#
conv_0_159311:
conv_0_159313:#
conv_1_159317:
conv_1_159319:#
conv_2_159323:
conv_2_159325:#
z_log_sig_159331:	?
z_log_sig_159333: 
z_mean_159336:	?
z_mean_159338:
identity

identity_1??conv_0/StatefulPartitionedCall?conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?!z_log_sig/StatefulPartitionedCall?z_mean/StatefulPartitionedCall?
conv_0/StatefulPartitionedCallStatefulPartitionedCallxconv_0_159311conv_0_159313*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_0_layer_call_and_return_conditional_losses_1589612 
conv_0/StatefulPartitionedCall?
!average_pooling1d/PartitionedCallPartitionedCall'conv_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_1589032#
!average_pooling1d/PartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0conv_1_159317conv_1_159319*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_1589832 
conv_1/StatefulPartitionedCall?
#average_pooling1d_1/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_1589182%
#average_pooling1d_1/PartitionedCall?
conv_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0conv_2_159323conv_2_159325*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_1590052 
conv_2/StatefulPartitionedCall?
#average_pooling1d_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *X
fSRQ
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_1589332%
#average_pooling1d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall,average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1590182
flatten/PartitionedCall?
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1591182
dropout/PartitionedCall?
!z_log_sig/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0z_log_sig_159331z_log_sig_159333*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_z_log_sig_layer_call_and_return_conditional_losses_1590372#
!z_log_sig/StatefulPartitionedCall?
z_mean/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0z_mean_159336z_mean_159338*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1590532 
z_mean/StatefulPartitionedCall?
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall"^z_log_sig/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*z_log_sig/StatefulPartitionedCall:output:0^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall"^z_log_sig/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????: : : : : : : : : : 2@
conv_0/StatefulPartitionedCallconv_0/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2F
!z_log_sig/StatefulPartitionedCall!z_log_sig/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:O K
,
_output_shapes
:??????????

_user_specified_nameX
?
?
B__inference_conv_0_layer_call_and_return_conditional_losses_159579

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_conv_2_layer_call_fn_159612

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_1590052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_dropout_layer_call_and_return_conditional_losses_159657

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

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
*__inference_z_log_sig_layer_call_fn_159685

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_z_log_sig_layer_call_and_return_conditional_losses_1590372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_conv_2_layer_call_and_return_conditional_losses_159627

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_z_log_sig_layer_call_and_return_conditional_losses_159695

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
k
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_158918

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
2	
AvgPool?
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_conv_0_layer_call_and_return_conditional_losses_158961

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159308
x#
conv_0_159276:
conv_0_159278:#
conv_1_159282:
conv_1_159284:#
conv_2_159288:
conv_2_159290:#
z_log_sig_159296:	?
z_log_sig_159298: 
z_mean_159301:	?
z_mean_159303:
identity

identity_1??conv_0/StatefulPartitionedCall?conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?!z_log_sig/StatefulPartitionedCall?z_mean/StatefulPartitionedCall?
conv_0/StatefulPartitionedCallStatefulPartitionedCallxconv_0_159276conv_0_159278*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_0_layer_call_and_return_conditional_losses_1589612 
conv_0/StatefulPartitionedCall?
!average_pooling1d/PartitionedCallPartitionedCall'conv_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_1589032#
!average_pooling1d/PartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0conv_1_159282conv_1_159284*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_1589832 
conv_1/StatefulPartitionedCall?
#average_pooling1d_1/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_1589182%
#average_pooling1d_1/PartitionedCall?
conv_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0conv_2_159288conv_2_159290*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_1590052 
conv_2/StatefulPartitionedCall?
#average_pooling1d_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *X
fSRQ
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_1589332%
#average_pooling1d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall,average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1590182
flatten/PartitionedCall?
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1590252
dropout/PartitionedCall?
!z_log_sig/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0z_log_sig_159296z_log_sig_159298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_z_log_sig_layer_call_and_return_conditional_losses_1590372#
!z_log_sig/StatefulPartitionedCall?
z_mean/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0z_mean_159301z_mean_159303*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1590532 
z_mean/StatefulPartitionedCall?
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall"^z_log_sig/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*z_log_sig/StatefulPartitionedCall:output:0^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall"^z_log_sig/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????: : : : : : : : : : 2@
conv_0/StatefulPartitionedCallconv_0/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2F
!z_log_sig/StatefulPartitionedCall!z_log_sig/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:O K
,
_output_shapes
:??????????

_user_specified_nameX
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_159638

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_z_mean_layer_call_fn_159666

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1590532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_CNN_encoder_layer_call_fn_159426

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7:	?
	unknown_8:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_1592212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
B__inference_z_mean_layer_call_and_return_conditional_losses_159676

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_conv_0_layer_call_fn_159564

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv_0_layer_call_and_return_conditional_losses_1589612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_flatten_layer_call_fn_159632

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1590182
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
k
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_158933

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
2	
AvgPool?
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
4
X/
serving_default_X:0??????????=
	z_log_sig0
StatefulPartitionedCall:0?????????:
z_mean0
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
?[
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
trainable_variables
regularization_losses
	variables
	keras_api

signatures
z_default_save_signature
{__call__
*|&call_and_return_all_conditional_losses"?X
_tf_keras_network?W{"name": "CNN_encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "CNN_encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3488, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "X"}, "name": "X", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv_0", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_0", "inbound_nodes": [[["X", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last"}, "name": "average_pooling1d", "inbound_nodes": [[["conv_0", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["average_pooling1d", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last"}, "name": "average_pooling1d_1", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["average_pooling1d_1", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last"}, "name": "average_pooling1d_2", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["average_pooling1d_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_sig", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_sig", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["X", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_sig", 0, 0]]}, "shared_object_id": 21, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3488, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3488, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3488, 1]}, "float32", "X"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CNN_encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3488, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "X"}, "name": "X", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "conv_0", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_0", "inbound_nodes": [[["X", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last"}, "name": "average_pooling1d", "inbound_nodes": [[["conv_0", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv1D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["average_pooling1d", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last"}, "name": "average_pooling1d_1", "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Conv1D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["average_pooling1d_1", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last"}, "name": "average_pooling1d_2", "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["average_pooling1d_2", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Dense", "config": {"name": "z_log_sig", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_sig", "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 20}], "input_layers": [["X", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_sig", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "X", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3488, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3488, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "X"}}
?


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv_0", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["X", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3488, 1]}}
?
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling1D", "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last"}, "inbound_nodes": [[["conv_0", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 24}}
?


kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["average_pooling1d", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 8}}, "shared_object_id": 25}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1742, 8]}}
?
!trainable_variables
"	variables
#regularization_losses
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "average_pooling1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last"}, "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 26}}
?


%kernel
&bias
'trainable_variables
(	variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["average_pooling1d_1", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 8}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 869, 8]}}
?
+trainable_variables
,	variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "average_pooling1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last"}, "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 28}}
?
/trainable_variables
0	variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["average_pooling1d_2", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 29}}
?
3trainable_variables
4	variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 14}
?	

7kernel
8bias
9trainable_variables
:	variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1732}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1732]}}
?	

=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "z_log_sig", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "z_log_sig", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1732}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1732]}}
f
0
1
2
3
%4
&5
76
87
=8
>9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
%4
&5
76
87
=8
>9"
trackable_list_wrapper
?
Cnon_trainable_variables
Dlayer_metrics
Elayer_regularization_losses
trainable_variables
Fmetrics
regularization_losses

Glayers
	variables
{__call__
z_default_save_signature
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
#:!2conv_0/kernel
:2conv_0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hnon_trainable_variables
Ilayer_metrics
Jlayer_regularization_losses
trainable_variables
	variables
Kmetrics

Llayers
regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Mnon_trainable_variables
Nlayer_metrics
Olayer_regularization_losses
trainable_variables
	variables
Pmetrics

Qlayers
regularization_losses
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!2conv_1/kernel
:2conv_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rnon_trainable_variables
Slayer_metrics
Tlayer_regularization_losses
trainable_variables
	variables
Umetrics

Vlayers
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wnon_trainable_variables
Xlayer_metrics
Ylayer_regularization_losses
!trainable_variables
"	variables
Zmetrics

[layers
#regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!2conv_2/kernel
:2conv_2/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables
]layer_metrics
^layer_regularization_losses
'trainable_variables
(	variables
_metrics

`layers
)regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
anon_trainable_variables
blayer_metrics
clayer_regularization_losses
+trainable_variables
,	variables
dmetrics

elayers
-regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables
glayer_metrics
hlayer_regularization_losses
/trainable_variables
0	variables
imetrics

jlayers
1regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
knon_trainable_variables
llayer_metrics
mlayer_regularization_losses
3trainable_variables
4	variables
nmetrics

olayers
5regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?2z_mean/kernel
:2z_mean/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables
qlayer_metrics
rlayer_regularization_losses
9trainable_variables
:	variables
smetrics

tlayers
;regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!	?2z_log_sig/kernel
:2z_log_sig/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
unon_trainable_variables
vlayer_metrics
wlayer_regularization_losses
?trainable_variables
@	variables
xmetrics

ylayers
Aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
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
!__inference__wrapped_model_158894?
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
annotations? *%?"
 ?
X??????????
?2?
,__inference_CNN_encoder_layer_call_fn_159086
,__inference_CNN_encoder_layer_call_fn_159399
,__inference_CNN_encoder_layer_call_fn_159426
,__inference_CNN_encoder_layer_call_fn_159273?
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
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159491
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159555
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159308
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159343?
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
'__inference_conv_0_layer_call_fn_159564?
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
B__inference_conv_0_layer_call_and_return_conditional_losses_159579?
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
2__inference_average_pooling1d_layer_call_fn_158909?
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
annotations? *3?0
.?+'???????????????????????????
?2?
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_158903?
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
annotations? *3?0
.?+'???????????????????????????
?2?
'__inference_conv_1_layer_call_fn_159588?
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
B__inference_conv_1_layer_call_and_return_conditional_losses_159603?
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
4__inference_average_pooling1d_1_layer_call_fn_158924?
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
annotations? *3?0
.?+'???????????????????????????
?2?
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_158918?
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
annotations? *3?0
.?+'???????????????????????????
?2?
'__inference_conv_2_layer_call_fn_159612?
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
B__inference_conv_2_layer_call_and_return_conditional_losses_159627?
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
4__inference_average_pooling1d_2_layer_call_fn_158939?
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
annotations? *3?0
.?+'???????????????????????????
?2?
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_158933?
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
annotations? *3?0
.?+'???????????????????????????
?2?
(__inference_flatten_layer_call_fn_159632?
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
C__inference_flatten_layer_call_and_return_conditional_losses_159638?
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
(__inference_dropout_layer_call_fn_159643
(__inference_dropout_layer_call_fn_159648?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_layer_call_and_return_conditional_losses_159653
C__inference_dropout_layer_call_and_return_conditional_losses_159657?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_z_mean_layer_call_fn_159666?
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
B__inference_z_mean_layer_call_and_return_conditional_losses_159676?
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
*__inference_z_log_sig_layer_call_fn_159685?
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
E__inference_z_log_sig_layer_call_and_return_conditional_losses_159695?
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
?B?
$__inference_signature_wrapper_159372X"?
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
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159308?
%&=>787?4
-?*
 ?
X??????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159343?
%&=>787?4
-?*
 ?
X??????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159491?
%&=>78<?9
2?/
%?"
inputs??????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
G__inference_CNN_encoder_layer_call_and_return_conditional_losses_159555?
%&=>78<?9
2?/
%?"
inputs??????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
,__inference_CNN_encoder_layer_call_fn_159086?
%&=>787?4
-?*
 ?
X??????????
p 

 
? "=?:
?
0?????????
?
1??????????
,__inference_CNN_encoder_layer_call_fn_159273?
%&=>787?4
-?*
 ?
X??????????
p

 
? "=?:
?
0?????????
?
1??????????
,__inference_CNN_encoder_layer_call_fn_159399?
%&=>78<?9
2?/
%?"
inputs??????????
p 

 
? "=?:
?
0?????????
?
1??????????
,__inference_CNN_encoder_layer_call_fn_159426?
%&=>78<?9
2?/
%?"
inputs??????????
p

 
? "=?:
?
0?????????
?
1??????????
!__inference__wrapped_model_158894?
%&=>78/?,
%?"
 ?
X??????????
? "a?^
0
	z_log_sig#? 
	z_log_sig?????????
*
z_mean ?
z_mean??????????
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_158918?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
4__inference_average_pooling1d_1_layer_call_fn_158924wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_158933?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
4__inference_average_pooling1d_2_layer_call_fn_158939wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_158903?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
2__inference_average_pooling1d_layer_call_fn_158909wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
B__inference_conv_0_layer_call_and_return_conditional_losses_159579f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
'__inference_conv_0_layer_call_fn_159564Y4?1
*?'
%?"
inputs??????????
? "????????????
B__inference_conv_1_layer_call_and_return_conditional_losses_159603f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
'__inference_conv_1_layer_call_fn_159588Y4?1
*?'
%?"
inputs??????????
? "????????????
B__inference_conv_2_layer_call_and_return_conditional_losses_159627f%&4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
'__inference_conv_2_layer_call_fn_159612Y%&4?1
*?'
%?"
inputs??????????
? "????????????
C__inference_dropout_layer_call_and_return_conditional_losses_159653^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_159657^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? }
(__inference_dropout_layer_call_fn_159643Q4?1
*?'
!?
inputs??????????
p 
? "???????????}
(__inference_dropout_layer_call_fn_159648Q4?1
*?'
!?
inputs??????????
p
? "????????????
C__inference_flatten_layer_call_and_return_conditional_losses_159638^4?1
*?'
%?"
inputs??????????
? "&?#
?
0??????????
? }
(__inference_flatten_layer_call_fn_159632Q4?1
*?'
%?"
inputs??????????
? "????????????
$__inference_signature_wrapper_159372?
%&=>784?1
? 
*?'
%
X ?
X??????????"a?^
0
	z_log_sig#? 
	z_log_sig?????????
*
z_mean ?
z_mean??????????
E__inference_z_log_sig_layer_call_and_return_conditional_losses_159695]=>0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ~
*__inference_z_log_sig_layer_call_fn_159685P=>0?-
&?#
!?
inputs??????????
? "???????????
B__inference_z_mean_layer_call_and_return_conditional_losses_159676]780?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_z_mean_layer_call_fn_159666P780?-
&?#
!?
inputs??????????
? "??????????