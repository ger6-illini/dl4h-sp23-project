��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.10.02unknown8��
�
Adam/lstm/lstm_cell_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/lstm/lstm_cell_19/bias/v
�
1Adam/lstm/lstm_cell_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell_19/bias/v*
_output_shapes
:@*
dtype0
�
)Adam/lstm/lstm_cell_19/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)Adam/lstm/lstm_cell_19/recurrent_kernel/v
�
=Adam/lstm/lstm_cell_19/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp)Adam/lstm/lstm_cell_19/recurrent_kernel/v*
_output_shapes

:@*
dtype0
�
Adam/lstm/lstm_cell_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*0
shared_name!Adam/lstm/lstm_cell_19/kernel/v
�
3Adam/lstm/lstm_cell_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell_19/kernel/v*
_output_shapes
:	�@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0
�
Adam/lstm/lstm_cell_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/lstm/lstm_cell_19/bias/m
�
1Adam/lstm/lstm_cell_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell_19/bias/m*
_output_shapes
:@*
dtype0
�
)Adam/lstm/lstm_cell_19/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)Adam/lstm/lstm_cell_19/recurrent_kernel/m
�
=Adam/lstm/lstm_cell_19/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp)Adam/lstm/lstm_cell_19/recurrent_kernel/m*
_output_shapes

:@*
dtype0
�
Adam/lstm/lstm_cell_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*0
shared_name!Adam/lstm/lstm_cell_19/kernel/m
�
3Adam/lstm/lstm_cell_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell_19/kernel/m*
_output_shapes
:	�@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
�
lstm/lstm_cell_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namelstm/lstm_cell_19/bias
}
*lstm/lstm_cell_19/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell_19/bias*
_output_shapes
:@*
dtype0
�
"lstm/lstm_cell_19/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"lstm/lstm_cell_19/recurrent_kernel
�
6lstm/lstm_cell_19/recurrent_kernel/Read/ReadVariableOpReadVariableOp"lstm/lstm_cell_19/recurrent_kernel*
_output_shapes

:@*
dtype0
�
lstm/lstm_cell_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*)
shared_namelstm/lstm_cell_19/kernel
�
,lstm/lstm_cell_19/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell_19/kernel*
_output_shapes
:	�@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
�
serving_default_lstm_inputPlaceholder*,
_output_shapes
:���������0�*
dtype0*!
shape:���������0�
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_inputlstm/lstm_cell_19/kernel"lstm/lstm_cell_19/recurrent_kernellstm/lstm_cell_19/biasdense/kernel
dense/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*1
config_proto!

CPU

GPU (2J 8� *.
f)R'
%__inference_signature_wrapper_1296080

NoOpNoOp
�-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�,
value�,B�, B�,
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
'
0
1
2
3
4*
'
0
1
2
3
4*
* 
�
 non_trainable_variables

!layers
"metrics
#layer_regularization_losses
$layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
%trace_0
&trace_1
'trace_2
(trace_3* 
6
)trace_0
*trace_1
+trace_2
,trace_3* 
* 
�
-iter

.beta_1

/beta_2
	0decay
1learning_ratemdmemfmgmhvivjvkvlvm*

2serving_default* 

0
1
2*

0
1
2*
* 
�

3states
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
9trace_0
:trace_1
;trace_2
<trace_3* 
6
=trace_0
>trace_1
?trace_2
@trace_3* 
* 
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator
H
state_size

kernel
recurrent_kernel
bias*
* 

0
1*

0
1*
* 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ntrace_0* 

Otrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUElstm/lstm_cell_19/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"lstm/lstm_cell_19/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUElstm/lstm_cell_19/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

P0
Q1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1
2*

0
1
2*
* 
�
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

Wtrace_0
Xtrace_1* 

Ytrace_0
Ztrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
[	variables
\	keras_api
	]total
	^count*
H
_	variables
`	keras_api
	atotal
	bcount
c
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 

]0
^1*

[	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

a0
b1*

_	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/lstm/lstm_cell_19/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE)Adam/lstm/lstm_cell_19/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/lstm/lstm_cell_19/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/lstm/lstm_cell_19/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE)Adam/lstm/lstm_cell_19/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/lstm/lstm_cell_19/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp,lstm/lstm_cell_19/kernel/Read/ReadVariableOp6lstm/lstm_cell_19/recurrent_kernel/Read/ReadVariableOp*lstm/lstm_cell_19/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp3Adam/lstm/lstm_cell_19/kernel/m/Read/ReadVariableOp=Adam/lstm/lstm_cell_19/recurrent_kernel/m/Read/ReadVariableOp1Adam/lstm/lstm_cell_19/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp3Adam/lstm/lstm_cell_19/kernel/v/Read/ReadVariableOp=Adam/lstm/lstm_cell_19/recurrent_kernel/v/Read/ReadVariableOp1Adam/lstm/lstm_cell_19/bias/v/Read/ReadVariableOpConst*%
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8� *)
f$R"
 __inference__traced_save_1297251
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biaslstm/lstm_cell_19/kernel"lstm/lstm_cell_19/recurrent_kernellstm/lstm_cell_19/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/lstm/lstm_cell_19/kernel/m)Adam/lstm/lstm_cell_19/recurrent_kernel/mAdam/lstm/lstm_cell_19/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/lstm/lstm_cell_19/kernel/v)Adam/lstm/lstm_cell_19/recurrent_kernel/vAdam/lstm/lstm_cell_19/bias/v*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8� *,
f'R%
#__inference__traced_restore_1297333��
�
�
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296041

lstm_input
lstm_1296028:	�@
lstm_1296030:@
lstm_1296032:@
dense_1296035:
dense_1296037:
identity��dense/StatefulPartitionedCall�lstm/StatefulPartitionedCall�
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_1296028lstm_1296030lstm_1296032*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1295748�
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_1296035dense_1296037*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1295767u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":���������0�: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:X T
,
_output_shapes
:���������0�
$
_user_specified_name
lstm_input
�9
�
A__inference_lstm_layer_call_and_return_conditional_losses_1295395

inputs'
lstm_cell_19_1295311:	�@&
lstm_cell_19_1295313:@"
lstm_cell_19_1295315:@
identity��$lstm_cell_19/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
$lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_19_1295311lstm_cell_19_1295313lstm_cell_19_1295315*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *R
fMRK
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1295310n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_19_1295311lstm_cell_19_1295313lstm_cell_19_1295315*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1295325*
condR
while_cond_1295324*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������u
NoOpNoOp%^lstm_cell_19/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2L
$lstm_cell_19/StatefulPartitionedCall$lstm_cell_19/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�K
�
A__inference_lstm_layer_call_and_return_conditional_losses_1296893

inputs>
+lstm_cell_19_matmul_readvariableop_resource:	�@?
-lstm_cell_19_matmul_1_readvariableop_resource:@:
,lstm_cell_19_biasadd_readvariableop_resource:@
identity��#lstm_cell_19/BiasAdd/ReadVariableOp�"lstm_cell_19/MatMul/ReadVariableOp�$lstm_cell_19/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:0����������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0�
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@^
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitn
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������p
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:���������w
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������h
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:���������{
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������p
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������e
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1296808*
condR
while_cond_1296807*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������0�: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:���������0�
 
_user_specified_nameinputs
�
�
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1297156

inputs
states_0
states_11
matmul_readvariableop_resource:	�@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������
"
_user_specified_name
states/1
�
�
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1295310

inputs

states
states_11
matmul_readvariableop_resource:	�@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates:OK
'
_output_shapes
:���������
 
_user_specified_namestates
�7
�

 __inference__traced_save_1297251
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop7
3savev2_lstm_lstm_cell_19_kernel_read_readvariableopA
=savev2_lstm_lstm_cell_19_recurrent_kernel_read_readvariableop5
1savev2_lstm_lstm_cell_19_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop>
:savev2_adam_lstm_lstm_cell_19_kernel_m_read_readvariableopH
Dsavev2_adam_lstm_lstm_cell_19_recurrent_kernel_m_read_readvariableop<
8savev2_adam_lstm_lstm_cell_19_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop>
:savev2_adam_lstm_lstm_cell_19_kernel_v_read_readvariableopH
Dsavev2_adam_lstm_lstm_cell_19_recurrent_kernel_v_read_readvariableop<
8savev2_adam_lstm_lstm_cell_19_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop3savev2_lstm_lstm_cell_19_kernel_read_readvariableop=savev2_lstm_lstm_cell_19_recurrent_kernel_read_readvariableop1savev2_lstm_lstm_cell_19_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop:savev2_adam_lstm_lstm_cell_19_kernel_m_read_readvariableopDsavev2_adam_lstm_lstm_cell_19_recurrent_kernel_m_read_readvariableop8savev2_adam_lstm_lstm_cell_19_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop:savev2_adam_lstm_lstm_cell_19_kernel_v_read_readvariableopDsavev2_adam_lstm_lstm_cell_19_recurrent_kernel_v_read_readvariableop8savev2_adam_lstm_lstm_cell_19_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :::	�@:@:@: : : : : : : : : :::	�@:@:@:::	�@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	�@:$ 

_output_shapes

:@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	�@:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	�@:$ 

_output_shapes

:@: 

_output_shapes
:@:

_output_shapes
: 
�$
�
while_body_1295325
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_19_1295349_0:	�@.
while_lstm_cell_19_1295351_0:@*
while_lstm_cell_19_1295353_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_19_1295349:	�@,
while_lstm_cell_19_1295351:@(
while_lstm_cell_19_1295353:@��*while/lstm_cell_19/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
*while/lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_19_1295349_0while_lstm_cell_19_1295351_0while_lstm_cell_19_1295353_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *R
fMRK
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1295310r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_19/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity3while/lstm_cell_19/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity3while/lstm_cell_19/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������y

while/NoOpNoOp+^while/lstm_cell_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_19_1295349while_lstm_cell_19_1295349_0":
while_lstm_cell_19_1295351while_lstm_cell_19_1295351_0":
while_lstm_cell_19_1295353while_lstm_cell_19_1295353_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_19/StatefulPartitionedCall*while/lstm_cell_19/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
<__inference_single_task_learning_model_layer_call_fn_1296110

inputs
unknown:	�@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*1
config_proto!

CPU

GPU (2J 8� *`
f[RY
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1295997o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":���������0�: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������0�
 
_user_specified_nameinputs
�
�
while_cond_1296952
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1296952___redundant_placeholder05
1while_while_cond_1296952___redundant_placeholder15
1while_while_cond_1296952___redundant_placeholder25
1while_while_cond_1296952___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
&__inference_lstm_layer_call_fn_1296436
inputs_0
unknown:	�@
	unknown_0:@
	unknown_1:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1295588o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�
�
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296057

lstm_input
lstm_1296044:	�@
lstm_1296046:@
lstm_1296048:@
dense_1296051:
dense_1296053:
identity��dense/StatefulPartitionedCall�lstm/StatefulPartitionedCall�
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_1296044lstm_1296046lstm_1296048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1295955�
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_1296051dense_1296053*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1295767u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":���������0�: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:X T
,
_output_shapes
:���������0�
$
_user_specified_name
lstm_input
�
�
2single_task_learning_model_lstm_while_cond_1295150\
Xsingle_task_learning_model_lstm_while_single_task_learning_model_lstm_while_loop_counterb
^single_task_learning_model_lstm_while_single_task_learning_model_lstm_while_maximum_iterations5
1single_task_learning_model_lstm_while_placeholder7
3single_task_learning_model_lstm_while_placeholder_17
3single_task_learning_model_lstm_while_placeholder_27
3single_task_learning_model_lstm_while_placeholder_3^
Zsingle_task_learning_model_lstm_while_less_single_task_learning_model_lstm_strided_slice_1u
qsingle_task_learning_model_lstm_while_single_task_learning_model_lstm_while_cond_1295150___redundant_placeholder0u
qsingle_task_learning_model_lstm_while_single_task_learning_model_lstm_while_cond_1295150___redundant_placeholder1u
qsingle_task_learning_model_lstm_while_single_task_learning_model_lstm_while_cond_1295150___redundant_placeholder2u
qsingle_task_learning_model_lstm_while_single_task_learning_model_lstm_while_cond_1295150___redundant_placeholder32
.single_task_learning_model_lstm_while_identity
�
*single_task_learning_model/lstm/while/LessLess1single_task_learning_model_lstm_while_placeholderZsingle_task_learning_model_lstm_while_less_single_task_learning_model_lstm_strided_slice_1*
T0*
_output_shapes
: �
.single_task_learning_model/lstm/while/IdentityIdentity.single_task_learning_model/lstm/while/Less:z:0*
T0
*
_output_shapes
: "i
.single_task_learning_model_lstm_while_identity7single_task_learning_model/lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
.__inference_lstm_cell_19_layer_call_fn_1297075

inputs
states_0
states_1
unknown:	�@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *R
fMRK
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1295310o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������
"
_user_specified_name
states/1
�

�
B__inference_dense_layer_call_and_return_conditional_losses_1295767

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_1296517
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1296517___redundant_placeholder05
1while_while_cond_1296517___redundant_placeholder15
1while_while_cond_1296517___redundant_placeholder25
1while_while_cond_1296517___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�K
�
A__inference_lstm_layer_call_and_return_conditional_losses_1296603
inputs_0>
+lstm_cell_19_matmul_readvariableop_resource:	�@?
-lstm_cell_19_matmul_1_readvariableop_resource:@:
,lstm_cell_19_biasadd_readvariableop_resource:@
identity��#lstm_cell_19/BiasAdd/ReadVariableOp�"lstm_cell_19/MatMul/ReadVariableOp�$lstm_cell_19/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0�
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@^
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitn
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������p
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:���������w
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������h
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:���������{
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������p
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������e
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1296518*
condR
while_cond_1296517*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�

�
B__inference_dense_layer_call_and_return_conditional_losses_1297058

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�K
�
A__inference_lstm_layer_call_and_return_conditional_losses_1295955

inputs>
+lstm_cell_19_matmul_readvariableop_resource:	�@?
-lstm_cell_19_matmul_1_readvariableop_resource:@:
,lstm_cell_19_biasadd_readvariableop_resource:@
identity��#lstm_cell_19/BiasAdd/ReadVariableOp�"lstm_cell_19/MatMul/ReadVariableOp�$lstm_cell_19/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:0����������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0�
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@^
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitn
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������p
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:���������w
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������h
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:���������{
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������p
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������e
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1295870*
condR
while_cond_1295869*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������0�: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:���������0�
 
_user_specified_nameinputs
�
�
while_cond_1295517
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1295517___redundant_placeholder05
1while_while_cond_1295517___redundant_placeholder15
1while_while_cond_1295517___redundant_placeholder25
1while_while_cond_1295517___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�K
�
A__inference_lstm_layer_call_and_return_conditional_losses_1296748
inputs_0>
+lstm_cell_19_matmul_readvariableop_resource:	�@?
-lstm_cell_19_matmul_1_readvariableop_resource:@:
,lstm_cell_19_biasadd_readvariableop_resource:@
identity��#lstm_cell_19/BiasAdd/ReadVariableOp�"lstm_cell_19/MatMul/ReadVariableOp�$lstm_cell_19/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0�
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@^
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitn
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������p
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:���������w
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������h
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:���������{
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������p
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������e
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1296663*
condR
while_cond_1296662*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�
�
<__inference_single_task_learning_model_layer_call_fn_1296095

inputs
unknown:	�@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*1
config_proto!

CPU

GPU (2J 8� *`
f[RY
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1295774o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":���������0�: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������0�
 
_user_specified_nameinputs
�?
�	
lstm_while_body_1296322&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_while_lstm_cell_19_matmul_readvariableop_resource_0:	�@L
:lstm_while_lstm_cell_19_matmul_1_readvariableop_resource_0:@G
9lstm_while_lstm_cell_19_biasadd_readvariableop_resource_0:@
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorI
6lstm_while_lstm_cell_19_matmul_readvariableop_resource:	�@J
8lstm_while_lstm_cell_19_matmul_1_readvariableop_resource:@E
7lstm_while_lstm_cell_19_biasadd_readvariableop_resource:@��.lstm/while/lstm_cell_19/BiasAdd/ReadVariableOp�-lstm/while/lstm_cell_19/MatMul/ReadVariableOp�/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp�
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
-lstm/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp8lstm_while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	�@*
dtype0�
lstm/while/lstm_cell_19/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:05lstm/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp:lstm_while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
 lstm/while/lstm_cell_19/MatMul_1MatMullstm_while_placeholder_27lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
lstm/while/lstm_cell_19/addAddV2(lstm/while/lstm_cell_19/MatMul:product:0*lstm/while/lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
.lstm/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp9lstm_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
lstm/while/lstm_cell_19/BiasAddBiasAddlstm/while/lstm_cell_19/add:z:06lstm/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@i
'lstm/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm/while/lstm_cell_19/splitSplit0lstm/while/lstm_cell_19/split/split_dim:output:0(lstm/while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm/while/lstm_cell_19/SigmoidSigmoid&lstm/while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:����������
!lstm/while/lstm_cell_19/Sigmoid_1Sigmoid&lstm/while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:����������
lstm/while/lstm_cell_19/mulMul%lstm/while/lstm_cell_19/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:���������~
lstm/while/lstm_cell_19/ReluRelu&lstm/while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
lstm/while/lstm_cell_19/mul_1Mul#lstm/while/lstm_cell_19/Sigmoid:y:0*lstm/while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm/while/lstm_cell_19/add_1AddV2lstm/while/lstm_cell_19/mul:z:0!lstm/while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:����������
!lstm/while/lstm_cell_19/Sigmoid_2Sigmoid&lstm/while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������{
lstm/while/lstm_cell_19/Relu_1Relu!lstm/while/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
lstm/while/lstm_cell_19/mul_2Mul%lstm/while/lstm_cell_19/Sigmoid_2:y:0,lstm/while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
5lstm/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1>lstm/while/TensorArrayV2Write/TensorListSetItem/index:output:0!lstm/while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:���R
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: �
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: �
lstm/while/Identity_4Identity!lstm/while/lstm_cell_19/mul_2:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:����������
lstm/while/Identity_5Identity!lstm/while/lstm_cell_19/add_1:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:����������
lstm/while/NoOpNoOp/^lstm/while/lstm_cell_19/BiasAdd/ReadVariableOp.^lstm/while/lstm_cell_19/MatMul/ReadVariableOp0^lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"t
7lstm_while_lstm_cell_19_biasadd_readvariableop_resource9lstm_while_lstm_cell_19_biasadd_readvariableop_resource_0"v
8lstm_while_lstm_cell_19_matmul_1_readvariableop_resource:lstm_while_lstm_cell_19_matmul_1_readvariableop_resource_0"r
6lstm_while_lstm_cell_19_matmul_readvariableop_resource8lstm_while_lstm_cell_19_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"�
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2`
.lstm/while/lstm_cell_19/BiasAdd/ReadVariableOp.lstm/while/lstm_cell_19/BiasAdd/ReadVariableOp2^
-lstm/while/lstm_cell_19/MatMul/ReadVariableOp-lstm/while/lstm_cell_19/MatMul/ReadVariableOp2b
/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
&__inference_lstm_layer_call_fn_1296458

inputs
unknown:	�@
	unknown_0:@
	unknown_1:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1295955o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������0�: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������0�
 
_user_specified_nameinputs
�K
�
A__inference_lstm_layer_call_and_return_conditional_losses_1297038

inputs>
+lstm_cell_19_matmul_readvariableop_resource:	�@?
-lstm_cell_19_matmul_1_readvariableop_resource:@:
,lstm_cell_19_biasadd_readvariableop_resource:@
identity��#lstm_cell_19/BiasAdd/ReadVariableOp�"lstm_cell_19/MatMul/ReadVariableOp�$lstm_cell_19/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:0����������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0�
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@^
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitn
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������p
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:���������w
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������h
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:���������{
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������p
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������e
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1296953*
condR
while_cond_1296952*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������0�: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:���������0�
 
_user_specified_nameinputs
�a
�
#__inference__traced_restore_1297333
file_prefix/
assignvariableop_dense_kernel:+
assignvariableop_1_dense_bias:>
+assignvariableop_2_lstm_lstm_cell_19_kernel:	�@G
5assignvariableop_3_lstm_lstm_cell_19_recurrent_kernel:@7
)assignvariableop_4_lstm_lstm_cell_19_bias:@&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_total: #
assignvariableop_13_count: 9
'assignvariableop_14_adam_dense_kernel_m:3
%assignvariableop_15_adam_dense_bias_m:F
3assignvariableop_16_adam_lstm_lstm_cell_19_kernel_m:	�@O
=assignvariableop_17_adam_lstm_lstm_cell_19_recurrent_kernel_m:@?
1assignvariableop_18_adam_lstm_lstm_cell_19_bias_m:@9
'assignvariableop_19_adam_dense_kernel_v:3
%assignvariableop_20_adam_dense_bias_v:F
3assignvariableop_21_adam_lstm_lstm_cell_19_kernel_v:	�@O
=assignvariableop_22_adam_lstm_lstm_cell_19_recurrent_kernel_v:@?
1assignvariableop_23_adam_lstm_lstm_cell_19_bias_v:@
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp+assignvariableop_2_lstm_lstm_cell_19_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp5assignvariableop_3_lstm_lstm_cell_19_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp)assignvariableop_4_lstm_lstm_cell_19_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_dense_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp%assignvariableop_15_adam_dense_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp3assignvariableop_16_adam_lstm_lstm_cell_19_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp=assignvariableop_17_adam_lstm_lstm_cell_19_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp1assignvariableop_18_adam_lstm_lstm_cell_19_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dense_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp3assignvariableop_21_adam_lstm_lstm_cell_19_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp=assignvariableop_22_adam_lstm_lstm_cell_19_recurrent_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_lstm_lstm_cell_19_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
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
�
�
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1295458

inputs

states
states_11
matmul_readvariableop_resource:	�@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates:OK
'
_output_shapes
:���������
 
_user_specified_namestates
�
�
%__inference_signature_wrapper_1296080

lstm_input
unknown:	�@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*1
config_proto!

CPU

GPU (2J 8� *+
f&R$
"__inference__wrapped_model_1295243o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":���������0�: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
,
_output_shapes
:���������0�
$
_user_specified_name
lstm_input
�
�
.__inference_lstm_cell_19_layer_call_fn_1297092

inputs
states_0
states_1
unknown:	�@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *R
fMRK
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1295458o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������
"
_user_specified_name
states/1
�9
�
while_body_1296518
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_19_matmul_readvariableop_resource_0:	�@G
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_19_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_19_matmul_readvariableop_resource:	�@E
3while_lstm_cell_19_matmul_1_readvariableop_resource:@@
2while_lstm_cell_19_biasadd_readvariableop_resource:@��)while/lstm_cell_19/BiasAdd/ReadVariableOp�(while/lstm_cell_19/MatMul/ReadVariableOp�*while/lstm_cell_19/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	�@*
dtype0�
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitz
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������|
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������t
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������|
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������q
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������y
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
&__inference_lstm_layer_call_fn_1296425
inputs_0
unknown:	�@
	unknown_0:@
	unknown_1:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1295395o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�?
�	
lstm_while_body_1296170&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_while_lstm_cell_19_matmul_readvariableop_resource_0:	�@L
:lstm_while_lstm_cell_19_matmul_1_readvariableop_resource_0:@G
9lstm_while_lstm_cell_19_biasadd_readvariableop_resource_0:@
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorI
6lstm_while_lstm_cell_19_matmul_readvariableop_resource:	�@J
8lstm_while_lstm_cell_19_matmul_1_readvariableop_resource:@E
7lstm_while_lstm_cell_19_biasadd_readvariableop_resource:@��.lstm/while/lstm_cell_19/BiasAdd/ReadVariableOp�-lstm/while/lstm_cell_19/MatMul/ReadVariableOp�/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp�
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
-lstm/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp8lstm_while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	�@*
dtype0�
lstm/while/lstm_cell_19/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:05lstm/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp:lstm_while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
 lstm/while/lstm_cell_19/MatMul_1MatMullstm_while_placeholder_27lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
lstm/while/lstm_cell_19/addAddV2(lstm/while/lstm_cell_19/MatMul:product:0*lstm/while/lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
.lstm/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp9lstm_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
lstm/while/lstm_cell_19/BiasAddBiasAddlstm/while/lstm_cell_19/add:z:06lstm/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@i
'lstm/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm/while/lstm_cell_19/splitSplit0lstm/while/lstm_cell_19/split/split_dim:output:0(lstm/while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm/while/lstm_cell_19/SigmoidSigmoid&lstm/while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:����������
!lstm/while/lstm_cell_19/Sigmoid_1Sigmoid&lstm/while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:����������
lstm/while/lstm_cell_19/mulMul%lstm/while/lstm_cell_19/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:���������~
lstm/while/lstm_cell_19/ReluRelu&lstm/while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
lstm/while/lstm_cell_19/mul_1Mul#lstm/while/lstm_cell_19/Sigmoid:y:0*lstm/while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm/while/lstm_cell_19/add_1AddV2lstm/while/lstm_cell_19/mul:z:0!lstm/while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:����������
!lstm/while/lstm_cell_19/Sigmoid_2Sigmoid&lstm/while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������{
lstm/while/lstm_cell_19/Relu_1Relu!lstm/while/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
lstm/while/lstm_cell_19/mul_2Mul%lstm/while/lstm_cell_19/Sigmoid_2:y:0,lstm/while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
5lstm/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1>lstm/while/TensorArrayV2Write/TensorListSetItem/index:output:0!lstm/while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:���R
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: �
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: �
lstm/while/Identity_4Identity!lstm/while/lstm_cell_19/mul_2:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:����������
lstm/while/Identity_5Identity!lstm/while/lstm_cell_19/add_1:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:����������
lstm/while/NoOpNoOp/^lstm/while/lstm_cell_19/BiasAdd/ReadVariableOp.^lstm/while/lstm_cell_19/MatMul/ReadVariableOp0^lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"t
7lstm_while_lstm_cell_19_biasadd_readvariableop_resource9lstm_while_lstm_cell_19_biasadd_readvariableop_resource_0"v
8lstm_while_lstm_cell_19_matmul_1_readvariableop_resource:lstm_while_lstm_cell_19_matmul_1_readvariableop_resource_0"r
6lstm_while_lstm_cell_19_matmul_readvariableop_resource8lstm_while_lstm_cell_19_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"�
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2`
.lstm/while/lstm_cell_19/BiasAdd/ReadVariableOp.lstm/while/lstm_cell_19/BiasAdd/ReadVariableOp2^
-lstm/while/lstm_cell_19/MatMul/ReadVariableOp-lstm/while/lstm_cell_19/MatMul/ReadVariableOp2b
/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_1295869
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1295869___redundant_placeholder05
1while_while_cond_1295869___redundant_placeholder15
1while_while_cond_1295869___redundant_placeholder25
1while_while_cond_1295869___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1295774

inputs
lstm_1295749:	�@
lstm_1295751:@
lstm_1295753:@
dense_1295768:
dense_1295770:
identity��dense/StatefulPartitionedCall�lstm/StatefulPartitionedCall�
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_1295749lstm_1295751lstm_1295753*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1295748�
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_1295768dense_1295770*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1295767u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":���������0�: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:T P
,
_output_shapes
:���������0�
 
_user_specified_nameinputs
�
�
while_cond_1295324
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1295324___redundant_placeholder05
1while_while_cond_1295324___redundant_placeholder15
1while_while_cond_1295324___redundant_placeholder25
1while_while_cond_1295324___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�Y
�
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296414

inputsC
0lstm_lstm_cell_19_matmul_readvariableop_resource:	�@D
2lstm_lstm_cell_19_matmul_1_readvariableop_resource:@?
1lstm_lstm_cell_19_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�(lstm/lstm_cell_19/BiasAdd/ReadVariableOp�'lstm/lstm_cell_19/MatMul/ReadVariableOp�)lstm/lstm_cell_19/MatMul_1/ReadVariableOp�
lstm/while@

lstm/ShapeShapeinputs*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:���������W
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*,
_output_shapes
:0����������N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���d
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
'lstm/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp0lstm_lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
lstm/lstm_cell_19/MatMulMatMullstm/strided_slice_2:output:0/lstm/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)lstm/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp2lstm_lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0�
lstm/lstm_cell_19/MatMul_1MatMullstm/zeros:output:01lstm/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
lstm/lstm_cell_19/addAddV2"lstm/lstm_cell_19/MatMul:product:0$lstm/lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
(lstm/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp1lstm_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
lstm/lstm_cell_19/BiasAddBiasAddlstm/lstm_cell_19/add:z:00lstm/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@c
!lstm/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm/lstm_cell_19/splitSplit*lstm/lstm_cell_19/split/split_dim:output:0"lstm/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitx
lstm/lstm_cell_19/SigmoidSigmoid lstm/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������z
lstm/lstm_cell_19/Sigmoid_1Sigmoid lstm/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:����������
lstm/lstm_cell_19/mulMullstm/lstm_cell_19/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:���������r
lstm/lstm_cell_19/ReluRelu lstm/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
lstm/lstm_cell_19/mul_1Mullstm/lstm_cell_19/Sigmoid:y:0$lstm/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm/lstm_cell_19/add_1AddV2lstm/lstm_cell_19/mul:z:0lstm/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������z
lstm/lstm_cell_19/Sigmoid_2Sigmoid lstm/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������o
lstm/lstm_cell_19/Relu_1Relulstm/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
lstm/lstm_cell_19/mul_2Mullstm/lstm_cell_19/Sigmoid_2:y:0&lstm/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   c
!lstm/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0*lstm/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���K
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������Y
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_lstm_cell_19_matmul_readvariableop_resource2lstm_lstm_cell_19_matmul_1_readvariableop_resource1lstm_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_while_body_1296322*#
condR
lstm_while_cond_1296321*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsm
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������f
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense/MatMulMatMullstm/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp)^lstm/lstm_cell_19/BiasAdd/ReadVariableOp(^lstm/lstm_cell_19/MatMul/ReadVariableOp*^lstm/lstm_cell_19/MatMul_1/ReadVariableOp^lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":���������0�: : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2T
(lstm/lstm_cell_19/BiasAdd/ReadVariableOp(lstm/lstm_cell_19/BiasAdd/ReadVariableOp2R
'lstm/lstm_cell_19/MatMul/ReadVariableOp'lstm/lstm_cell_19/MatMul/ReadVariableOp2V
)lstm/lstm_cell_19/MatMul_1/ReadVariableOp)lstm/lstm_cell_19/MatMul_1/ReadVariableOp2

lstm/while
lstm/while:T P
,
_output_shapes
:���������0�
 
_user_specified_nameinputs
�9
�
while_body_1295663
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_19_matmul_readvariableop_resource_0:	�@G
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_19_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_19_matmul_readvariableop_resource:	�@E
3while_lstm_cell_19_matmul_1_readvariableop_resource:@@
2while_lstm_cell_19_biasadd_readvariableop_resource:@��)while/lstm_cell_19/BiasAdd/ReadVariableOp�(while/lstm_cell_19/MatMul/ReadVariableOp�*while/lstm_cell_19/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	�@*
dtype0�
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitz
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������|
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������t
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������|
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������q
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������y
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�	
�
lstm_while_cond_1296321&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1?
;lstm_while_lstm_while_cond_1296321___redundant_placeholder0?
;lstm_while_lstm_while_cond_1296321___redundant_placeholder1?
;lstm_while_lstm_while_cond_1296321___redundant_placeholder2?
;lstm_while_lstm_while_cond_1296321___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_1296662
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1296662___redundant_placeholder05
1while_while_cond_1296662___redundant_placeholder15
1while_while_cond_1296662___redundant_placeholder25
1while_while_cond_1296662___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�$
�
while_body_1295518
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_19_1295542_0:	�@.
while_lstm_cell_19_1295544_0:@*
while_lstm_cell_19_1295546_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_19_1295542:	�@,
while_lstm_cell_19_1295544:@(
while_lstm_cell_19_1295546:@��*while/lstm_cell_19/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
*while/lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_19_1295542_0while_lstm_cell_19_1295544_0while_lstm_cell_19_1295546_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *R
fMRK
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1295458r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_19/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity3while/lstm_cell_19/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity3while/lstm_cell_19/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������y

while/NoOpNoOp+^while/lstm_cell_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_19_1295542while_lstm_cell_19_1295542_0":
while_lstm_cell_19_1295544while_lstm_cell_19_1295544_0":
while_lstm_cell_19_1295546while_lstm_cell_19_1295546_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_19/StatefulPartitionedCall*while/lstm_cell_19/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�9
�
while_body_1296953
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_19_matmul_readvariableop_resource_0:	�@G
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_19_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_19_matmul_readvariableop_resource:	�@E
3while_lstm_cell_19_matmul_1_readvariableop_resource:@@
2while_lstm_cell_19_biasadd_readvariableop_resource:@��)while/lstm_cell_19/BiasAdd/ReadVariableOp�(while/lstm_cell_19/MatMul/ReadVariableOp�*while/lstm_cell_19/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	�@*
dtype0�
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitz
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������|
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������t
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������|
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������q
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������y
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�	
�
<__inference_single_task_learning_model_layer_call_fn_1296025

lstm_input
unknown:	�@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*1
config_proto!

CPU

GPU (2J 8� *`
f[RY
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1295997o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":���������0�: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
,
_output_shapes
:���������0�
$
_user_specified_name
lstm_input
�9
�
while_body_1295870
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_19_matmul_readvariableop_resource_0:	�@G
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_19_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_19_matmul_readvariableop_resource:	�@E
3while_lstm_cell_19_matmul_1_readvariableop_resource:@@
2while_lstm_cell_19_biasadd_readvariableop_resource:@��)while/lstm_cell_19/BiasAdd/ReadVariableOp�(while/lstm_cell_19/MatMul/ReadVariableOp�*while/lstm_cell_19/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	�@*
dtype0�
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitz
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������|
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������t
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������|
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������q
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������y
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�9
�
while_body_1296663
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_19_matmul_readvariableop_resource_0:	�@G
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_19_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_19_matmul_readvariableop_resource:	�@E
3while_lstm_cell_19_matmul_1_readvariableop_resource:@@
2while_lstm_cell_19_biasadd_readvariableop_resource:@��)while/lstm_cell_19/BiasAdd/ReadVariableOp�(while/lstm_cell_19/MatMul/ReadVariableOp�*while/lstm_cell_19/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	�@*
dtype0�
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitz
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������|
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������t
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������|
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������q
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������y
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�_
�
2single_task_learning_model_lstm_while_body_1295151\
Xsingle_task_learning_model_lstm_while_single_task_learning_model_lstm_while_loop_counterb
^single_task_learning_model_lstm_while_single_task_learning_model_lstm_while_maximum_iterations5
1single_task_learning_model_lstm_while_placeholder7
3single_task_learning_model_lstm_while_placeholder_17
3single_task_learning_model_lstm_while_placeholder_27
3single_task_learning_model_lstm_while_placeholder_3[
Wsingle_task_learning_model_lstm_while_single_task_learning_model_lstm_strided_slice_1_0�
�single_task_learning_model_lstm_while_tensorarrayv2read_tensorlistgetitem_single_task_learning_model_lstm_tensorarrayunstack_tensorlistfromtensor_0f
Ssingle_task_learning_model_lstm_while_lstm_cell_19_matmul_readvariableop_resource_0:	�@g
Usingle_task_learning_model_lstm_while_lstm_cell_19_matmul_1_readvariableop_resource_0:@b
Tsingle_task_learning_model_lstm_while_lstm_cell_19_biasadd_readvariableop_resource_0:@2
.single_task_learning_model_lstm_while_identity4
0single_task_learning_model_lstm_while_identity_14
0single_task_learning_model_lstm_while_identity_24
0single_task_learning_model_lstm_while_identity_34
0single_task_learning_model_lstm_while_identity_44
0single_task_learning_model_lstm_while_identity_5Y
Usingle_task_learning_model_lstm_while_single_task_learning_model_lstm_strided_slice_1�
�single_task_learning_model_lstm_while_tensorarrayv2read_tensorlistgetitem_single_task_learning_model_lstm_tensorarrayunstack_tensorlistfromtensord
Qsingle_task_learning_model_lstm_while_lstm_cell_19_matmul_readvariableop_resource:	�@e
Ssingle_task_learning_model_lstm_while_lstm_cell_19_matmul_1_readvariableop_resource:@`
Rsingle_task_learning_model_lstm_while_lstm_cell_19_biasadd_readvariableop_resource:@��Isingle_task_learning_model/lstm/while/lstm_cell_19/BiasAdd/ReadVariableOp�Hsingle_task_learning_model/lstm/while/lstm_cell_19/MatMul/ReadVariableOp�Jsingle_task_learning_model/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp�
Wsingle_task_learning_model/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Isingle_task_learning_model/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�single_task_learning_model_lstm_while_tensorarrayv2read_tensorlistgetitem_single_task_learning_model_lstm_tensorarrayunstack_tensorlistfromtensor_01single_task_learning_model_lstm_while_placeholder`single_task_learning_model/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
Hsingle_task_learning_model/lstm/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOpSsingle_task_learning_model_lstm_while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	�@*
dtype0�
9single_task_learning_model/lstm/while/lstm_cell_19/MatMulMatMulPsingle_task_learning_model/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0Psingle_task_learning_model/lstm/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Jsingle_task_learning_model/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOpUsingle_task_learning_model_lstm_while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
;single_task_learning_model/lstm/while/lstm_cell_19/MatMul_1MatMul3single_task_learning_model_lstm_while_placeholder_2Rsingle_task_learning_model/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
6single_task_learning_model/lstm/while/lstm_cell_19/addAddV2Csingle_task_learning_model/lstm/while/lstm_cell_19/MatMul:product:0Esingle_task_learning_model/lstm/while/lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
Isingle_task_learning_model/lstm/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOpTsingle_task_learning_model_lstm_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
:single_task_learning_model/lstm/while/lstm_cell_19/BiasAddBiasAdd:single_task_learning_model/lstm/while/lstm_cell_19/add:z:0Qsingle_task_learning_model/lstm/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Bsingle_task_learning_model/lstm/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
8single_task_learning_model/lstm/while/lstm_cell_19/splitSplitKsingle_task_learning_model/lstm/while/lstm_cell_19/split/split_dim:output:0Csingle_task_learning_model/lstm/while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
:single_task_learning_model/lstm/while/lstm_cell_19/SigmoidSigmoidAsingle_task_learning_model/lstm/while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:����������
<single_task_learning_model/lstm/while/lstm_cell_19/Sigmoid_1SigmoidAsingle_task_learning_model/lstm/while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:����������
6single_task_learning_model/lstm/while/lstm_cell_19/mulMul@single_task_learning_model/lstm/while/lstm_cell_19/Sigmoid_1:y:03single_task_learning_model_lstm_while_placeholder_3*
T0*'
_output_shapes
:����������
7single_task_learning_model/lstm/while/lstm_cell_19/ReluReluAsingle_task_learning_model/lstm/while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
8single_task_learning_model/lstm/while/lstm_cell_19/mul_1Mul>single_task_learning_model/lstm/while/lstm_cell_19/Sigmoid:y:0Esingle_task_learning_model/lstm/while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:����������
8single_task_learning_model/lstm/while/lstm_cell_19/add_1AddV2:single_task_learning_model/lstm/while/lstm_cell_19/mul:z:0<single_task_learning_model/lstm/while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:����������
<single_task_learning_model/lstm/while/lstm_cell_19/Sigmoid_2SigmoidAsingle_task_learning_model/lstm/while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:����������
9single_task_learning_model/lstm/while/lstm_cell_19/Relu_1Relu<single_task_learning_model/lstm/while/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
8single_task_learning_model/lstm/while/lstm_cell_19/mul_2Mul@single_task_learning_model/lstm/while/lstm_cell_19/Sigmoid_2:y:0Gsingle_task_learning_model/lstm/while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:����������
Psingle_task_learning_model/lstm/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
Jsingle_task_learning_model/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem3single_task_learning_model_lstm_while_placeholder_1Ysingle_task_learning_model/lstm/while/TensorArrayV2Write/TensorListSetItem/index:output:0<single_task_learning_model/lstm/while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:���m
+single_task_learning_model/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
)single_task_learning_model/lstm/while/addAddV21single_task_learning_model_lstm_while_placeholder4single_task_learning_model/lstm/while/add/y:output:0*
T0*
_output_shapes
: o
-single_task_learning_model/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
+single_task_learning_model/lstm/while/add_1AddV2Xsingle_task_learning_model_lstm_while_single_task_learning_model_lstm_while_loop_counter6single_task_learning_model/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: �
.single_task_learning_model/lstm/while/IdentityIdentity/single_task_learning_model/lstm/while/add_1:z:0+^single_task_learning_model/lstm/while/NoOp*
T0*
_output_shapes
: �
0single_task_learning_model/lstm/while/Identity_1Identity^single_task_learning_model_lstm_while_single_task_learning_model_lstm_while_maximum_iterations+^single_task_learning_model/lstm/while/NoOp*
T0*
_output_shapes
: �
0single_task_learning_model/lstm/while/Identity_2Identity-single_task_learning_model/lstm/while/add:z:0+^single_task_learning_model/lstm/while/NoOp*
T0*
_output_shapes
: �
0single_task_learning_model/lstm/while/Identity_3IdentityZsingle_task_learning_model/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^single_task_learning_model/lstm/while/NoOp*
T0*
_output_shapes
: �
0single_task_learning_model/lstm/while/Identity_4Identity<single_task_learning_model/lstm/while/lstm_cell_19/mul_2:z:0+^single_task_learning_model/lstm/while/NoOp*
T0*'
_output_shapes
:����������
0single_task_learning_model/lstm/while/Identity_5Identity<single_task_learning_model/lstm/while/lstm_cell_19/add_1:z:0+^single_task_learning_model/lstm/while/NoOp*
T0*'
_output_shapes
:����������
*single_task_learning_model/lstm/while/NoOpNoOpJ^single_task_learning_model/lstm/while/lstm_cell_19/BiasAdd/ReadVariableOpI^single_task_learning_model/lstm/while/lstm_cell_19/MatMul/ReadVariableOpK^single_task_learning_model/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "i
.single_task_learning_model_lstm_while_identity7single_task_learning_model/lstm/while/Identity:output:0"m
0single_task_learning_model_lstm_while_identity_19single_task_learning_model/lstm/while/Identity_1:output:0"m
0single_task_learning_model_lstm_while_identity_29single_task_learning_model/lstm/while/Identity_2:output:0"m
0single_task_learning_model_lstm_while_identity_39single_task_learning_model/lstm/while/Identity_3:output:0"m
0single_task_learning_model_lstm_while_identity_49single_task_learning_model/lstm/while/Identity_4:output:0"m
0single_task_learning_model_lstm_while_identity_59single_task_learning_model/lstm/while/Identity_5:output:0"�
Rsingle_task_learning_model_lstm_while_lstm_cell_19_biasadd_readvariableop_resourceTsingle_task_learning_model_lstm_while_lstm_cell_19_biasadd_readvariableop_resource_0"�
Ssingle_task_learning_model_lstm_while_lstm_cell_19_matmul_1_readvariableop_resourceUsingle_task_learning_model_lstm_while_lstm_cell_19_matmul_1_readvariableop_resource_0"�
Qsingle_task_learning_model_lstm_while_lstm_cell_19_matmul_readvariableop_resourceSsingle_task_learning_model_lstm_while_lstm_cell_19_matmul_readvariableop_resource_0"�
Usingle_task_learning_model_lstm_while_single_task_learning_model_lstm_strided_slice_1Wsingle_task_learning_model_lstm_while_single_task_learning_model_lstm_strided_slice_1_0"�
�single_task_learning_model_lstm_while_tensorarrayv2read_tensorlistgetitem_single_task_learning_model_lstm_tensorarrayunstack_tensorlistfromtensor�single_task_learning_model_lstm_while_tensorarrayv2read_tensorlistgetitem_single_task_learning_model_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2�
Isingle_task_learning_model/lstm/while/lstm_cell_19/BiasAdd/ReadVariableOpIsingle_task_learning_model/lstm/while/lstm_cell_19/BiasAdd/ReadVariableOp2�
Hsingle_task_learning_model/lstm/while/lstm_cell_19/MatMul/ReadVariableOpHsingle_task_learning_model/lstm/while/lstm_cell_19/MatMul/ReadVariableOp2�
Jsingle_task_learning_model/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOpJsingle_task_learning_model/lstm/while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1295997

inputs
lstm_1295984:	�@
lstm_1295986:@
lstm_1295988:@
dense_1295991:
dense_1295993:
identity��dense/StatefulPartitionedCall�lstm/StatefulPartitionedCall�
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_1295984lstm_1295986lstm_1295988*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1295955�
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_1295991dense_1295993*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1295767u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":���������0�: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:T P
,
_output_shapes
:���������0�
 
_user_specified_nameinputs
�
�
while_cond_1296807
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1296807___redundant_placeholder05
1while_while_cond_1296807___redundant_placeholder15
1while_while_cond_1296807___redundant_placeholder25
1while_while_cond_1296807___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�9
�
while_body_1296808
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_19_matmul_readvariableop_resource_0:	�@G
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_19_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_19_matmul_readvariableop_resource:	�@E
3while_lstm_cell_19_matmul_1_readvariableop_resource:@@
2while_lstm_cell_19_biasadd_readvariableop_resource:@��)while/lstm_cell_19/BiasAdd/ReadVariableOp�(while/lstm_cell_19/MatMul/ReadVariableOp�*while/lstm_cell_19/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	�@*
dtype0�
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitz
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������|
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������t
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������|
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������q
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������y
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�Y
�
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296262

inputsC
0lstm_lstm_cell_19_matmul_readvariableop_resource:	�@D
2lstm_lstm_cell_19_matmul_1_readvariableop_resource:@?
1lstm_lstm_cell_19_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�(lstm/lstm_cell_19/BiasAdd/ReadVariableOp�'lstm/lstm_cell_19/MatMul/ReadVariableOp�)lstm/lstm_cell_19/MatMul_1/ReadVariableOp�
lstm/while@

lstm/ShapeShapeinputs*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:���������W
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*,
_output_shapes
:0����������N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���d
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
'lstm/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp0lstm_lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
lstm/lstm_cell_19/MatMulMatMullstm/strided_slice_2:output:0/lstm/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)lstm/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp2lstm_lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0�
lstm/lstm_cell_19/MatMul_1MatMullstm/zeros:output:01lstm/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
lstm/lstm_cell_19/addAddV2"lstm/lstm_cell_19/MatMul:product:0$lstm/lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
(lstm/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp1lstm_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
lstm/lstm_cell_19/BiasAddBiasAddlstm/lstm_cell_19/add:z:00lstm/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@c
!lstm/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm/lstm_cell_19/splitSplit*lstm/lstm_cell_19/split/split_dim:output:0"lstm/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitx
lstm/lstm_cell_19/SigmoidSigmoid lstm/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������z
lstm/lstm_cell_19/Sigmoid_1Sigmoid lstm/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:����������
lstm/lstm_cell_19/mulMullstm/lstm_cell_19/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:���������r
lstm/lstm_cell_19/ReluRelu lstm/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
lstm/lstm_cell_19/mul_1Mullstm/lstm_cell_19/Sigmoid:y:0$lstm/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm/lstm_cell_19/add_1AddV2lstm/lstm_cell_19/mul:z:0lstm/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������z
lstm/lstm_cell_19/Sigmoid_2Sigmoid lstm/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������o
lstm/lstm_cell_19/Relu_1Relulstm/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
lstm/lstm_cell_19/mul_2Mullstm/lstm_cell_19/Sigmoid_2:y:0&lstm/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   c
!lstm/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0*lstm/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���K
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������Y
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_lstm_cell_19_matmul_readvariableop_resource2lstm_lstm_cell_19_matmul_1_readvariableop_resource1lstm_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_while_body_1296170*#
condR
lstm_while_cond_1296169*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsm
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������f
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense/MatMulMatMullstm/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp)^lstm/lstm_cell_19/BiasAdd/ReadVariableOp(^lstm/lstm_cell_19/MatMul/ReadVariableOp*^lstm/lstm_cell_19/MatMul_1/ReadVariableOp^lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":���������0�: : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2T
(lstm/lstm_cell_19/BiasAdd/ReadVariableOp(lstm/lstm_cell_19/BiasAdd/ReadVariableOp2R
'lstm/lstm_cell_19/MatMul/ReadVariableOp'lstm/lstm_cell_19/MatMul/ReadVariableOp2V
)lstm/lstm_cell_19/MatMul_1/ReadVariableOp)lstm/lstm_cell_19/MatMul_1/ReadVariableOp2

lstm/while
lstm/while:T P
,
_output_shapes
:���������0�
 
_user_specified_nameinputs
�
�
"__inference__wrapped_model_1295243

lstm_input^
Ksingle_task_learning_model_lstm_lstm_cell_19_matmul_readvariableop_resource:	�@_
Msingle_task_learning_model_lstm_lstm_cell_19_matmul_1_readvariableop_resource:@Z
Lsingle_task_learning_model_lstm_lstm_cell_19_biasadd_readvariableop_resource:@Q
?single_task_learning_model_dense_matmul_readvariableop_resource:N
@single_task_learning_model_dense_biasadd_readvariableop_resource:
identity��7single_task_learning_model/dense/BiasAdd/ReadVariableOp�6single_task_learning_model/dense/MatMul/ReadVariableOp�Csingle_task_learning_model/lstm/lstm_cell_19/BiasAdd/ReadVariableOp�Bsingle_task_learning_model/lstm/lstm_cell_19/MatMul/ReadVariableOp�Dsingle_task_learning_model/lstm/lstm_cell_19/MatMul_1/ReadVariableOp�%single_task_learning_model/lstm/while_
%single_task_learning_model/lstm/ShapeShape
lstm_input*
T0*
_output_shapes
:}
3single_task_learning_model/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5single_task_learning_model/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5single_task_learning_model/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-single_task_learning_model/lstm/strided_sliceStridedSlice.single_task_learning_model/lstm/Shape:output:0<single_task_learning_model/lstm/strided_slice/stack:output:0>single_task_learning_model/lstm/strided_slice/stack_1:output:0>single_task_learning_model/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.single_task_learning_model/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
,single_task_learning_model/lstm/zeros/packedPack6single_task_learning_model/lstm/strided_slice:output:07single_task_learning_model/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:p
+single_task_learning_model/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%single_task_learning_model/lstm/zerosFill5single_task_learning_model/lstm/zeros/packed:output:04single_task_learning_model/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:���������r
0single_task_learning_model/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
.single_task_learning_model/lstm/zeros_1/packedPack6single_task_learning_model/lstm/strided_slice:output:09single_task_learning_model/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:r
-single_task_learning_model/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'single_task_learning_model/lstm/zeros_1Fill7single_task_learning_model/lstm/zeros_1/packed:output:06single_task_learning_model/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:����������
.single_task_learning_model/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
)single_task_learning_model/lstm/transpose	Transpose
lstm_input7single_task_learning_model/lstm/transpose/perm:output:0*
T0*,
_output_shapes
:0�����������
'single_task_learning_model/lstm/Shape_1Shape-single_task_learning_model/lstm/transpose:y:0*
T0*
_output_shapes
:
5single_task_learning_model/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
7single_task_learning_model/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
7single_task_learning_model/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/single_task_learning_model/lstm/strided_slice_1StridedSlice0single_task_learning_model/lstm/Shape_1:output:0>single_task_learning_model/lstm/strided_slice_1/stack:output:0@single_task_learning_model/lstm/strided_slice_1/stack_1:output:0@single_task_learning_model/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
;single_task_learning_model/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
-single_task_learning_model/lstm/TensorArrayV2TensorListReserveDsingle_task_learning_model/lstm/TensorArrayV2/element_shape:output:08single_task_learning_model/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Usingle_task_learning_model/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Gsingle_task_learning_model/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor-single_task_learning_model/lstm/transpose:y:0^single_task_learning_model/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���
5single_task_learning_model/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
7single_task_learning_model/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
7single_task_learning_model/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/single_task_learning_model/lstm/strided_slice_2StridedSlice-single_task_learning_model/lstm/transpose:y:0>single_task_learning_model/lstm/strided_slice_2/stack:output:0@single_task_learning_model/lstm/strided_slice_2/stack_1:output:0@single_task_learning_model/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
Bsingle_task_learning_model/lstm/lstm_cell_19/MatMul/ReadVariableOpReadVariableOpKsingle_task_learning_model_lstm_lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
3single_task_learning_model/lstm/lstm_cell_19/MatMulMatMul8single_task_learning_model/lstm/strided_slice_2:output:0Jsingle_task_learning_model/lstm/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Dsingle_task_learning_model/lstm/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOpMsingle_task_learning_model_lstm_lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0�
5single_task_learning_model/lstm/lstm_cell_19/MatMul_1MatMul.single_task_learning_model/lstm/zeros:output:0Lsingle_task_learning_model/lstm/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0single_task_learning_model/lstm/lstm_cell_19/addAddV2=single_task_learning_model/lstm/lstm_cell_19/MatMul:product:0?single_task_learning_model/lstm/lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
Csingle_task_learning_model/lstm/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOpLsingle_task_learning_model_lstm_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
4single_task_learning_model/lstm/lstm_cell_19/BiasAddBiasAdd4single_task_learning_model/lstm/lstm_cell_19/add:z:0Ksingle_task_learning_model/lstm/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
<single_task_learning_model/lstm/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
2single_task_learning_model/lstm/lstm_cell_19/splitSplitEsingle_task_learning_model/lstm/lstm_cell_19/split/split_dim:output:0=single_task_learning_model/lstm/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
4single_task_learning_model/lstm/lstm_cell_19/SigmoidSigmoid;single_task_learning_model/lstm/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:����������
6single_task_learning_model/lstm/lstm_cell_19/Sigmoid_1Sigmoid;single_task_learning_model/lstm/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:����������
0single_task_learning_model/lstm/lstm_cell_19/mulMul:single_task_learning_model/lstm/lstm_cell_19/Sigmoid_1:y:00single_task_learning_model/lstm/zeros_1:output:0*
T0*'
_output_shapes
:����������
1single_task_learning_model/lstm/lstm_cell_19/ReluRelu;single_task_learning_model/lstm/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
2single_task_learning_model/lstm/lstm_cell_19/mul_1Mul8single_task_learning_model/lstm/lstm_cell_19/Sigmoid:y:0?single_task_learning_model/lstm/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:����������
2single_task_learning_model/lstm/lstm_cell_19/add_1AddV24single_task_learning_model/lstm/lstm_cell_19/mul:z:06single_task_learning_model/lstm/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:����������
6single_task_learning_model/lstm/lstm_cell_19/Sigmoid_2Sigmoid;single_task_learning_model/lstm/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:����������
3single_task_learning_model/lstm/lstm_cell_19/Relu_1Relu6single_task_learning_model/lstm/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
2single_task_learning_model/lstm/lstm_cell_19/mul_2Mul:single_task_learning_model/lstm/lstm_cell_19/Sigmoid_2:y:0Asingle_task_learning_model/lstm/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:����������
=single_task_learning_model/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ~
<single_task_learning_model/lstm/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
/single_task_learning_model/lstm/TensorArrayV2_1TensorListReserveFsingle_task_learning_model/lstm/TensorArrayV2_1/element_shape:output:0Esingle_task_learning_model/lstm/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���f
$single_task_learning_model/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : �
8single_task_learning_model/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������t
2single_task_learning_model/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �	
%single_task_learning_model/lstm/whileWhile;single_task_learning_model/lstm/while/loop_counter:output:0Asingle_task_learning_model/lstm/while/maximum_iterations:output:0-single_task_learning_model/lstm/time:output:08single_task_learning_model/lstm/TensorArrayV2_1:handle:0.single_task_learning_model/lstm/zeros:output:00single_task_learning_model/lstm/zeros_1:output:08single_task_learning_model/lstm/strided_slice_1:output:0Wsingle_task_learning_model/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ksingle_task_learning_model_lstm_lstm_cell_19_matmul_readvariableop_resourceMsingle_task_learning_model_lstm_lstm_cell_19_matmul_1_readvariableop_resourceLsingle_task_learning_model_lstm_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *>
body6R4
2single_task_learning_model_lstm_while_body_1295151*>
cond6R4
2single_task_learning_model_lstm_while_cond_1295150*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
Psingle_task_learning_model/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Bsingle_task_learning_model/lstm/TensorArrayV2Stack/TensorListStackTensorListStack.single_task_learning_model/lstm/while:output:3Ysingle_task_learning_model/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elements�
5single_task_learning_model/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
7single_task_learning_model/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
7single_task_learning_model/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/single_task_learning_model/lstm/strided_slice_3StridedSliceKsingle_task_learning_model/lstm/TensorArrayV2Stack/TensorListStack:tensor:0>single_task_learning_model/lstm/strided_slice_3/stack:output:0@single_task_learning_model/lstm/strided_slice_3/stack_1:output:0@single_task_learning_model/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
0single_task_learning_model/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
+single_task_learning_model/lstm/transpose_1	TransposeKsingle_task_learning_model/lstm/TensorArrayV2Stack/TensorListStack:tensor:09single_task_learning_model/lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������{
'single_task_learning_model/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
6single_task_learning_model/dense/MatMul/ReadVariableOpReadVariableOp?single_task_learning_model_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
'single_task_learning_model/dense/MatMulMatMul8single_task_learning_model/lstm/strided_slice_3:output:0>single_task_learning_model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
7single_task_learning_model/dense/BiasAdd/ReadVariableOpReadVariableOp@single_task_learning_model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
(single_task_learning_model/dense/BiasAddBiasAdd1single_task_learning_model/dense/MatMul:product:0?single_task_learning_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(single_task_learning_model/dense/SigmoidSigmoid1single_task_learning_model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������{
IdentityIdentity,single_task_learning_model/dense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp8^single_task_learning_model/dense/BiasAdd/ReadVariableOp7^single_task_learning_model/dense/MatMul/ReadVariableOpD^single_task_learning_model/lstm/lstm_cell_19/BiasAdd/ReadVariableOpC^single_task_learning_model/lstm/lstm_cell_19/MatMul/ReadVariableOpE^single_task_learning_model/lstm/lstm_cell_19/MatMul_1/ReadVariableOp&^single_task_learning_model/lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":���������0�: : : : : 2r
7single_task_learning_model/dense/BiasAdd/ReadVariableOp7single_task_learning_model/dense/BiasAdd/ReadVariableOp2p
6single_task_learning_model/dense/MatMul/ReadVariableOp6single_task_learning_model/dense/MatMul/ReadVariableOp2�
Csingle_task_learning_model/lstm/lstm_cell_19/BiasAdd/ReadVariableOpCsingle_task_learning_model/lstm/lstm_cell_19/BiasAdd/ReadVariableOp2�
Bsingle_task_learning_model/lstm/lstm_cell_19/MatMul/ReadVariableOpBsingle_task_learning_model/lstm/lstm_cell_19/MatMul/ReadVariableOp2�
Dsingle_task_learning_model/lstm/lstm_cell_19/MatMul_1/ReadVariableOpDsingle_task_learning_model/lstm/lstm_cell_19/MatMul_1/ReadVariableOp2N
%single_task_learning_model/lstm/while%single_task_learning_model/lstm/while:X T
,
_output_shapes
:���������0�
$
_user_specified_name
lstm_input
�K
�
A__inference_lstm_layer_call_and_return_conditional_losses_1295748

inputs>
+lstm_cell_19_matmul_readvariableop_resource:	�@?
-lstm_cell_19_matmul_1_readvariableop_resource:@:
,lstm_cell_19_biasadd_readvariableop_resource:@
identity��#lstm_cell_19/BiasAdd/ReadVariableOp�"lstm_cell_19/MatMul/ReadVariableOp�$lstm_cell_19/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:0����������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0�
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@^
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitn
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:���������p
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:���������w
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������h
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:���������{
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:���������p
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:���������e
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1295663*
condR
while_cond_1295662*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������0�: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:���������0�
 
_user_specified_nameinputs
�
�
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1297124

inputs
states_0
states_11
matmul_readvariableop_resource:	�@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������
"
_user_specified_name
states/1
�
�
&__inference_lstm_layer_call_fn_1296447

inputs
unknown:	�@
	unknown_0:@
	unknown_1:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1295748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������0�: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������0�
 
_user_specified_nameinputs
�	
�
lstm_while_cond_1296169&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1?
;lstm_while_lstm_while_cond_1296169___redundant_placeholder0?
;lstm_while_lstm_while_cond_1296169___redundant_placeholder1?
;lstm_while_lstm_while_cond_1296169___redundant_placeholder2?
;lstm_while_lstm_while_cond_1296169___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
'__inference_dense_layer_call_fn_1297047

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1295767o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
A__inference_lstm_layer_call_and_return_conditional_losses_1295588

inputs'
lstm_cell_19_1295504:	�@&
lstm_cell_19_1295506:@"
lstm_cell_19_1295508:@
identity��$lstm_cell_19/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
$lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_19_1295504lstm_cell_19_1295506lstm_cell_19_1295508*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8� *R
fMRK
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1295458n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_19_1295504lstm_cell_19_1295506lstm_cell_19_1295508*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1295518*
condR
while_cond_1295517*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������u
NoOpNoOp%^lstm_cell_19/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2L
$lstm_cell_19/StatefulPartitionedCall$lstm_cell_19/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
while_cond_1295662
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1295662___redundant_placeholder05
1while_while_cond_1295662___redundant_placeholder15
1while_while_cond_1295662___redundant_placeholder25
1while_while_cond_1295662___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�	
�
<__inference_single_task_learning_model_layer_call_fn_1295787

lstm_input
unknown:	�@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*1
config_proto!

CPU

GPU (2J 8� *`
f[RY
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1295774o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":���������0�: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
,
_output_shapes
:���������0�
$
_user_specified_name
lstm_input"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
F

lstm_input8
serving_default_lstm_input:0���������0�9
dense0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 non_trainable_variables

!layers
"metrics
#layer_regularization_losses
$layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
%trace_0
&trace_1
'trace_2
(trace_32�
<__inference_single_task_learning_model_layer_call_fn_1295787
<__inference_single_task_learning_model_layer_call_fn_1296095
<__inference_single_task_learning_model_layer_call_fn_1296110
<__inference_single_task_learning_model_layer_call_fn_1296025�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z%trace_0z&trace_1z'trace_2z(trace_3
�
)trace_0
*trace_1
+trace_2
,trace_32�
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296262
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296414
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296041
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296057�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z)trace_0z*trace_1z+trace_2z,trace_3
�B�
"__inference__wrapped_model_1295243
lstm_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
-iter

.beta_1

/beta_2
	0decay
1learning_ratemdmemfmgmhvivjvkvlvm"
	optimizer
,
2serving_default"
signature_map
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

3states
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
9trace_0
:trace_1
;trace_2
<trace_32�
&__inference_lstm_layer_call_fn_1296425
&__inference_lstm_layer_call_fn_1296436
&__inference_lstm_layer_call_fn_1296447
&__inference_lstm_layer_call_fn_1296458�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z9trace_0z:trace_1z;trace_2z<trace_3
�
=trace_0
>trace_1
?trace_2
@trace_32�
A__inference_lstm_layer_call_and_return_conditional_losses_1296603
A__inference_lstm_layer_call_and_return_conditional_losses_1296748
A__inference_lstm_layer_call_and_return_conditional_losses_1296893
A__inference_lstm_layer_call_and_return_conditional_losses_1297038�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z=trace_0z>trace_1z?trace_2z@trace_3
"
_generic_user_object
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator
H
state_size

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
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
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ntrace_02�
'__inference_dense_layer_call_fn_1297047�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zNtrace_0
�
Otrace_02�
B__inference_dense_layer_call_and_return_conditional_losses_1297058�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zOtrace_0
:2dense/kernel
:2
dense/bias
+:)	�@2lstm/lstm_cell_19/kernel
4:2@2"lstm/lstm_cell_19/recurrent_kernel
$:"@2lstm/lstm_cell_19/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
<__inference_single_task_learning_model_layer_call_fn_1295787
lstm_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
<__inference_single_task_learning_model_layer_call_fn_1296095inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
<__inference_single_task_learning_model_layer_call_fn_1296110inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
<__inference_single_task_learning_model_layer_call_fn_1296025
lstm_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296262inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296414inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296041
lstm_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296057
lstm_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
%__inference_signature_wrapper_1296080
lstm_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_lstm_layer_call_fn_1296425inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_lstm_layer_call_fn_1296436inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_lstm_layer_call_fn_1296447inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_lstm_layer_call_fn_1296458inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_lstm_layer_call_and_return_conditional_losses_1296603inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_lstm_layer_call_and_return_conditional_losses_1296748inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_lstm_layer_call_and_return_conditional_losses_1296893inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_lstm_layer_call_and_return_conditional_losses_1297038inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
Wtrace_0
Xtrace_12�
.__inference_lstm_cell_19_layer_call_fn_1297075
.__inference_lstm_cell_19_layer_call_fn_1297092�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zWtrace_0zXtrace_1
�
Ytrace_0
Ztrace_12�
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1297124
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1297156�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zYtrace_0zZtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
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
�B�
'__inference_dense_layer_call_fn_1297047inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_layer_call_and_return_conditional_losses_1297058inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
[	variables
\	keras_api
	]total
	^count"
_tf_keras_metric
^
_	variables
`	keras_api
	atotal
	bcount
c
_fn_kwargs"
_tf_keras_metric
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
�B�
.__inference_lstm_cell_19_layer_call_fn_1297075inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_lstm_cell_19_layer_call_fn_1297092inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1297124inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1297156inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
]0
^1"
trackable_list_wrapper
-
[	variables"
_generic_user_object
:  (2total
:  (2count
.
a0
b1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
0:.	�@2Adam/lstm/lstm_cell_19/kernel/m
9:7@2)Adam/lstm/lstm_cell_19/recurrent_kernel/m
):'@2Adam/lstm/lstm_cell_19/bias/m
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
0:.	�@2Adam/lstm/lstm_cell_19/kernel/v
9:7@2)Adam/lstm/lstm_cell_19/recurrent_kernel/v
):'@2Adam/lstm/lstm_cell_19/bias/v�
"__inference__wrapped_model_1295243p8�5
.�+
)�&

lstm_input���������0�
� "-�*
(
dense�
dense����������
B__inference_dense_layer_call_and_return_conditional_losses_1297058\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
'__inference_dense_layer_call_fn_1297047O/�,
%�"
 �
inputs���������
� "�����������
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1297124���~
w�t
!�
inputs����������
K�H
"�
states/0���������
"�
states/1���������
p 
� "s�p
i�f
�
0/0���������
E�B
�
0/1/0���������
�
0/1/1���������
� �
I__inference_lstm_cell_19_layer_call_and_return_conditional_losses_1297156���~
w�t
!�
inputs����������
K�H
"�
states/0���������
"�
states/1���������
p
� "s�p
i�f
�
0/0���������
E�B
�
0/1/0���������
�
0/1/1���������
� �
.__inference_lstm_cell_19_layer_call_fn_1297075���~
w�t
!�
inputs����������
K�H
"�
states/0���������
"�
states/1���������
p 
� "c�`
�
0���������
A�>
�
1/0���������
�
1/1����������
.__inference_lstm_cell_19_layer_call_fn_1297092���~
w�t
!�
inputs����������
K�H
"�
states/0���������
"�
states/1���������
p
� "c�`
�
0���������
A�>
�
1/0���������
�
1/1����������
A__inference_lstm_layer_call_and_return_conditional_losses_1296603~P�M
F�C
5�2
0�-
inputs/0�������������������

 
p 

 
� "%�"
�
0���������
� �
A__inference_lstm_layer_call_and_return_conditional_losses_1296748~P�M
F�C
5�2
0�-
inputs/0�������������������

 
p

 
� "%�"
�
0���������
� �
A__inference_lstm_layer_call_and_return_conditional_losses_1296893n@�=
6�3
%�"
inputs���������0�

 
p 

 
� "%�"
�
0���������
� �
A__inference_lstm_layer_call_and_return_conditional_losses_1297038n@�=
6�3
%�"
inputs���������0�

 
p

 
� "%�"
�
0���������
� �
&__inference_lstm_layer_call_fn_1296425qP�M
F�C
5�2
0�-
inputs/0�������������������

 
p 

 
� "�����������
&__inference_lstm_layer_call_fn_1296436qP�M
F�C
5�2
0�-
inputs/0�������������������

 
p

 
� "�����������
&__inference_lstm_layer_call_fn_1296447a@�=
6�3
%�"
inputs���������0�

 
p 

 
� "�����������
&__inference_lstm_layer_call_fn_1296458a@�=
6�3
%�"
inputs���������0�

 
p

 
� "�����������
%__inference_signature_wrapper_1296080~F�C
� 
<�9
7

lstm_input)�&

lstm_input���������0�"-�*
(
dense�
dense����������
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296041p@�=
6�3
)�&

lstm_input���������0�
p 

 
� "%�"
�
0���������
� �
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296057p@�=
6�3
)�&

lstm_input���������0�
p

 
� "%�"
�
0���������
� �
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296262l<�9
2�/
%�"
inputs���������0�
p 

 
� "%�"
�
0���������
� �
W__inference_single_task_learning_model_layer_call_and_return_conditional_losses_1296414l<�9
2�/
%�"
inputs���������0�
p

 
� "%�"
�
0���������
� �
<__inference_single_task_learning_model_layer_call_fn_1295787c@�=
6�3
)�&

lstm_input���������0�
p 

 
� "�����������
<__inference_single_task_learning_model_layer_call_fn_1296025c@�=
6�3
)�&

lstm_input���������0�
p

 
� "�����������
<__inference_single_task_learning_model_layer_call_fn_1296095_<�9
2�/
%�"
inputs���������0�
p 

 
� "�����������
<__inference_single_task_learning_model_layer_call_fn_1296110_<�9
2�/
%�"
inputs���������0�
p

 
� "����������