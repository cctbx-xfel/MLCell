ߛ%
��
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
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
delete_old_dirsbool(�
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
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.02v2.9.0-rc2-42-g8a20d54a3c18�!
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:2*
dtype0
�
Adam/reg_norm_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/reg_norm_1/beta/v
}
*Adam/reg_norm_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/reg_norm_1/beta/v*
_output_shapes
:2*
dtype0
�
Adam/reg_norm_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*(
shared_nameAdam/reg_norm_1/gamma/v

+Adam/reg_norm_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/reg_norm_1/gamma/v*
_output_shapes
:2*
dtype0
�
Adam/regress_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/regress_1/bias/v
{
)Adam/regress_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/regress_1/bias/v*
_output_shapes
:2*
dtype0
�
Adam/regress_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/regress_1/kernel/v
�
+Adam/regress_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/regress_1/kernel/v*
_output_shapes

:2*
dtype0
�
Adam/ebottleneck/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/ebottleneck/bias/v

+Adam/ebottleneck/bias/v/Read/ReadVariableOpReadVariableOpAdam/ebottleneck/bias/v*
_output_shapes
:*
dtype0
�
Adam/ebottleneck/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P**
shared_nameAdam/ebottleneck/kernel/v
�
-Adam/ebottleneck/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ebottleneck/kernel/v*
_output_shapes

:P*
dtype0
�
Adam/elnorm_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*%
shared_nameAdam/elnorm_3/beta/v
y
(Adam/elnorm_3/beta/v/Read/ReadVariableOpReadVariableOpAdam/elnorm_3/beta/v*
_output_shapes
:P*
dtype0
�
Adam/elnorm_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/elnorm_3/gamma/v
{
)Adam/elnorm_3/gamma/v/Read/ReadVariableOpReadVariableOpAdam/elnorm_3/gamma/v*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameAdam/ebatch_norm_3/beta/v
�
-Adam/ebatch_norm_3/beta/v/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_3/beta/v*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/ebatch_norm_3/gamma/v
�
.Adam/ebatch_norm_3/gamma/v/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_3/gamma/v*
_output_shapes
:P*
dtype0
�
Adam/encoder_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/encoder_3/bias/v
{
)Adam/encoder_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/encoder_3/bias/v*
_output_shapes
:P*
dtype0
�
Adam/encoder_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*(
shared_nameAdam/encoder_3/kernel/v
�
+Adam/encoder_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encoder_3/kernel/v*
_output_shapes

:PP*
dtype0
�
Adam/elnorm_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*%
shared_nameAdam/elnorm_2/beta/v
y
(Adam/elnorm_2/beta/v/Read/ReadVariableOpReadVariableOpAdam/elnorm_2/beta/v*
_output_shapes
:P*
dtype0
�
Adam/elnorm_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/elnorm_2/gamma/v
{
)Adam/elnorm_2/gamma/v/Read/ReadVariableOpReadVariableOpAdam/elnorm_2/gamma/v*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameAdam/ebatch_norm_2/beta/v
�
-Adam/ebatch_norm_2/beta/v/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_2/beta/v*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/ebatch_norm_2/gamma/v
�
.Adam/ebatch_norm_2/gamma/v/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_2/gamma/v*
_output_shapes
:P*
dtype0
�
Adam/encoder_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/encoder_2/bias/v
{
)Adam/encoder_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/encoder_2/bias/v*
_output_shapes
:P*
dtype0
�
Adam/encoder_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*(
shared_nameAdam/encoder_2/kernel/v
�
+Adam/encoder_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encoder_2/kernel/v*
_output_shapes

:PP*
dtype0
�
Adam/elnorm_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*%
shared_nameAdam/elnorm_1/beta/v
y
(Adam/elnorm_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/elnorm_1/beta/v*
_output_shapes
:P*
dtype0
�
Adam/elnorm_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/elnorm_1/gamma/v
{
)Adam/elnorm_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/elnorm_1/gamma/v*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameAdam/ebatch_norm_1/beta/v
�
-Adam/ebatch_norm_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_1/beta/v*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/ebatch_norm_1/gamma/v
�
.Adam/ebatch_norm_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_1/gamma/v*
_output_shapes
:P*
dtype0
�
Adam/encoder_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/encoder_1/bias/v
{
)Adam/encoder_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/encoder_1/bias/v*
_output_shapes
:P*
dtype0
�
Adam/encoder_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*(
shared_nameAdam/encoder_1/kernel/v
�
+Adam/encoder_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encoder_1/kernel/v*
_output_shapes

:PP*
dtype0
�
Adam/elnorm_0/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*%
shared_nameAdam/elnorm_0/beta/v
y
(Adam/elnorm_0/beta/v/Read/ReadVariableOpReadVariableOpAdam/elnorm_0/beta/v*
_output_shapes
:P*
dtype0
�
Adam/elnorm_0/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/elnorm_0/gamma/v
{
)Adam/elnorm_0/gamma/v/Read/ReadVariableOpReadVariableOpAdam/elnorm_0/gamma/v*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_0/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameAdam/ebatch_norm_0/beta/v
�
-Adam/ebatch_norm_0/beta/v/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_0/beta/v*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_0/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/ebatch_norm_0/gamma/v
�
.Adam/ebatch_norm_0/gamma/v/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_0/gamma/v*
_output_shapes
:P*
dtype0
�
Adam/encoder_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/encoder_0/bias/v
{
)Adam/encoder_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/encoder_0/bias/v*
_output_shapes
:P*
dtype0
�
Adam/encoder_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*(
shared_nameAdam/encoder_0/kernel/v
�
+Adam/encoder_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/encoder_0/kernel/v*
_output_shapes

:P*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:2*
dtype0
�
Adam/reg_norm_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/reg_norm_1/beta/m
}
*Adam/reg_norm_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/reg_norm_1/beta/m*
_output_shapes
:2*
dtype0
�
Adam/reg_norm_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*(
shared_nameAdam/reg_norm_1/gamma/m

+Adam/reg_norm_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/reg_norm_1/gamma/m*
_output_shapes
:2*
dtype0
�
Adam/regress_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/regress_1/bias/m
{
)Adam/regress_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/regress_1/bias/m*
_output_shapes
:2*
dtype0
�
Adam/regress_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/regress_1/kernel/m
�
+Adam/regress_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/regress_1/kernel/m*
_output_shapes

:2*
dtype0
�
Adam/ebottleneck/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/ebottleneck/bias/m

+Adam/ebottleneck/bias/m/Read/ReadVariableOpReadVariableOpAdam/ebottleneck/bias/m*
_output_shapes
:*
dtype0
�
Adam/ebottleneck/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P**
shared_nameAdam/ebottleneck/kernel/m
�
-Adam/ebottleneck/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ebottleneck/kernel/m*
_output_shapes

:P*
dtype0
�
Adam/elnorm_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*%
shared_nameAdam/elnorm_3/beta/m
y
(Adam/elnorm_3/beta/m/Read/ReadVariableOpReadVariableOpAdam/elnorm_3/beta/m*
_output_shapes
:P*
dtype0
�
Adam/elnorm_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/elnorm_3/gamma/m
{
)Adam/elnorm_3/gamma/m/Read/ReadVariableOpReadVariableOpAdam/elnorm_3/gamma/m*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameAdam/ebatch_norm_3/beta/m
�
-Adam/ebatch_norm_3/beta/m/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_3/beta/m*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/ebatch_norm_3/gamma/m
�
.Adam/ebatch_norm_3/gamma/m/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_3/gamma/m*
_output_shapes
:P*
dtype0
�
Adam/encoder_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/encoder_3/bias/m
{
)Adam/encoder_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/encoder_3/bias/m*
_output_shapes
:P*
dtype0
�
Adam/encoder_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*(
shared_nameAdam/encoder_3/kernel/m
�
+Adam/encoder_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encoder_3/kernel/m*
_output_shapes

:PP*
dtype0
�
Adam/elnorm_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*%
shared_nameAdam/elnorm_2/beta/m
y
(Adam/elnorm_2/beta/m/Read/ReadVariableOpReadVariableOpAdam/elnorm_2/beta/m*
_output_shapes
:P*
dtype0
�
Adam/elnorm_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/elnorm_2/gamma/m
{
)Adam/elnorm_2/gamma/m/Read/ReadVariableOpReadVariableOpAdam/elnorm_2/gamma/m*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameAdam/ebatch_norm_2/beta/m
�
-Adam/ebatch_norm_2/beta/m/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_2/beta/m*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/ebatch_norm_2/gamma/m
�
.Adam/ebatch_norm_2/gamma/m/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_2/gamma/m*
_output_shapes
:P*
dtype0
�
Adam/encoder_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/encoder_2/bias/m
{
)Adam/encoder_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/encoder_2/bias/m*
_output_shapes
:P*
dtype0
�
Adam/encoder_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*(
shared_nameAdam/encoder_2/kernel/m
�
+Adam/encoder_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encoder_2/kernel/m*
_output_shapes

:PP*
dtype0
�
Adam/elnorm_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*%
shared_nameAdam/elnorm_1/beta/m
y
(Adam/elnorm_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/elnorm_1/beta/m*
_output_shapes
:P*
dtype0
�
Adam/elnorm_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/elnorm_1/gamma/m
{
)Adam/elnorm_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/elnorm_1/gamma/m*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameAdam/ebatch_norm_1/beta/m
�
-Adam/ebatch_norm_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_1/beta/m*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/ebatch_norm_1/gamma/m
�
.Adam/ebatch_norm_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_1/gamma/m*
_output_shapes
:P*
dtype0
�
Adam/encoder_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/encoder_1/bias/m
{
)Adam/encoder_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/encoder_1/bias/m*
_output_shapes
:P*
dtype0
�
Adam/encoder_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*(
shared_nameAdam/encoder_1/kernel/m
�
+Adam/encoder_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encoder_1/kernel/m*
_output_shapes

:PP*
dtype0
�
Adam/elnorm_0/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*%
shared_nameAdam/elnorm_0/beta/m
y
(Adam/elnorm_0/beta/m/Read/ReadVariableOpReadVariableOpAdam/elnorm_0/beta/m*
_output_shapes
:P*
dtype0
�
Adam/elnorm_0/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/elnorm_0/gamma/m
{
)Adam/elnorm_0/gamma/m/Read/ReadVariableOpReadVariableOpAdam/elnorm_0/gamma/m*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_0/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameAdam/ebatch_norm_0/beta/m
�
-Adam/ebatch_norm_0/beta/m/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_0/beta/m*
_output_shapes
:P*
dtype0
�
Adam/ebatch_norm_0/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/ebatch_norm_0/gamma/m
�
.Adam/ebatch_norm_0/gamma/m/Read/ReadVariableOpReadVariableOpAdam/ebatch_norm_0/gamma/m*
_output_shapes
:P*
dtype0
�
Adam/encoder_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*&
shared_nameAdam/encoder_0/bias/m
{
)Adam/encoder_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/encoder_0/bias/m*
_output_shapes
:P*
dtype0
�
Adam/encoder_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*(
shared_nameAdam/encoder_0/kernel/m
�
+Adam/encoder_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/encoder_0/kernel/m*
_output_shapes

:P*
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
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:2*
dtype0
�
reg_norm_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*+
shared_namereg_norm_1/moving_variance
�
.reg_norm_1/moving_variance/Read/ReadVariableOpReadVariableOpreg_norm_1/moving_variance*
_output_shapes
:2*
dtype0
�
reg_norm_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_namereg_norm_1/moving_mean
}
*reg_norm_1/moving_mean/Read/ReadVariableOpReadVariableOpreg_norm_1/moving_mean*
_output_shapes
:2*
dtype0
v
reg_norm_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2* 
shared_namereg_norm_1/beta
o
#reg_norm_1/beta/Read/ReadVariableOpReadVariableOpreg_norm_1/beta*
_output_shapes
:2*
dtype0
x
reg_norm_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*!
shared_namereg_norm_1/gamma
q
$reg_norm_1/gamma/Read/ReadVariableOpReadVariableOpreg_norm_1/gamma*
_output_shapes
:2*
dtype0
t
regress_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_nameregress_1/bias
m
"regress_1/bias/Read/ReadVariableOpReadVariableOpregress_1/bias*
_output_shapes
:2*
dtype0
|
regress_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*!
shared_nameregress_1/kernel
u
$regress_1/kernel/Read/ReadVariableOpReadVariableOpregress_1/kernel*
_output_shapes

:2*
dtype0
x
ebottleneck/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameebottleneck/bias
q
$ebottleneck/bias/Read/ReadVariableOpReadVariableOpebottleneck/bias*
_output_shapes
:*
dtype0
�
ebottleneck/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*#
shared_nameebottleneck/kernel
y
&ebottleneck/kernel/Read/ReadVariableOpReadVariableOpebottleneck/kernel*
_output_shapes

:P*
dtype0
r
elnorm_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameelnorm_3/beta
k
!elnorm_3/beta/Read/ReadVariableOpReadVariableOpelnorm_3/beta*
_output_shapes
:P*
dtype0
t
elnorm_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameelnorm_3/gamma
m
"elnorm_3/gamma/Read/ReadVariableOpReadVariableOpelnorm_3/gamma*
_output_shapes
:P*
dtype0
�
ebatch_norm_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*.
shared_nameebatch_norm_3/moving_variance
�
1ebatch_norm_3/moving_variance/Read/ReadVariableOpReadVariableOpebatch_norm_3/moving_variance*
_output_shapes
:P*
dtype0
�
ebatch_norm_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameebatch_norm_3/moving_mean
�
-ebatch_norm_3/moving_mean/Read/ReadVariableOpReadVariableOpebatch_norm_3/moving_mean*
_output_shapes
:P*
dtype0
|
ebatch_norm_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*#
shared_nameebatch_norm_3/beta
u
&ebatch_norm_3/beta/Read/ReadVariableOpReadVariableOpebatch_norm_3/beta*
_output_shapes
:P*
dtype0
~
ebatch_norm_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameebatch_norm_3/gamma
w
'ebatch_norm_3/gamma/Read/ReadVariableOpReadVariableOpebatch_norm_3/gamma*
_output_shapes
:P*
dtype0
t
encoder_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameencoder_3/bias
m
"encoder_3/bias/Read/ReadVariableOpReadVariableOpencoder_3/bias*
_output_shapes
:P*
dtype0
|
encoder_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*!
shared_nameencoder_3/kernel
u
$encoder_3/kernel/Read/ReadVariableOpReadVariableOpencoder_3/kernel*
_output_shapes

:PP*
dtype0
r
elnorm_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameelnorm_2/beta
k
!elnorm_2/beta/Read/ReadVariableOpReadVariableOpelnorm_2/beta*
_output_shapes
:P*
dtype0
t
elnorm_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameelnorm_2/gamma
m
"elnorm_2/gamma/Read/ReadVariableOpReadVariableOpelnorm_2/gamma*
_output_shapes
:P*
dtype0
�
ebatch_norm_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*.
shared_nameebatch_norm_2/moving_variance
�
1ebatch_norm_2/moving_variance/Read/ReadVariableOpReadVariableOpebatch_norm_2/moving_variance*
_output_shapes
:P*
dtype0
�
ebatch_norm_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameebatch_norm_2/moving_mean
�
-ebatch_norm_2/moving_mean/Read/ReadVariableOpReadVariableOpebatch_norm_2/moving_mean*
_output_shapes
:P*
dtype0
|
ebatch_norm_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*#
shared_nameebatch_norm_2/beta
u
&ebatch_norm_2/beta/Read/ReadVariableOpReadVariableOpebatch_norm_2/beta*
_output_shapes
:P*
dtype0
~
ebatch_norm_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameebatch_norm_2/gamma
w
'ebatch_norm_2/gamma/Read/ReadVariableOpReadVariableOpebatch_norm_2/gamma*
_output_shapes
:P*
dtype0
t
encoder_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameencoder_2/bias
m
"encoder_2/bias/Read/ReadVariableOpReadVariableOpencoder_2/bias*
_output_shapes
:P*
dtype0
|
encoder_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*!
shared_nameencoder_2/kernel
u
$encoder_2/kernel/Read/ReadVariableOpReadVariableOpencoder_2/kernel*
_output_shapes

:PP*
dtype0
r
elnorm_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameelnorm_1/beta
k
!elnorm_1/beta/Read/ReadVariableOpReadVariableOpelnorm_1/beta*
_output_shapes
:P*
dtype0
t
elnorm_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameelnorm_1/gamma
m
"elnorm_1/gamma/Read/ReadVariableOpReadVariableOpelnorm_1/gamma*
_output_shapes
:P*
dtype0
�
ebatch_norm_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*.
shared_nameebatch_norm_1/moving_variance
�
1ebatch_norm_1/moving_variance/Read/ReadVariableOpReadVariableOpebatch_norm_1/moving_variance*
_output_shapes
:P*
dtype0
�
ebatch_norm_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameebatch_norm_1/moving_mean
�
-ebatch_norm_1/moving_mean/Read/ReadVariableOpReadVariableOpebatch_norm_1/moving_mean*
_output_shapes
:P*
dtype0
|
ebatch_norm_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*#
shared_nameebatch_norm_1/beta
u
&ebatch_norm_1/beta/Read/ReadVariableOpReadVariableOpebatch_norm_1/beta*
_output_shapes
:P*
dtype0
~
ebatch_norm_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameebatch_norm_1/gamma
w
'ebatch_norm_1/gamma/Read/ReadVariableOpReadVariableOpebatch_norm_1/gamma*
_output_shapes
:P*
dtype0
t
encoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameencoder_1/bias
m
"encoder_1/bias/Read/ReadVariableOpReadVariableOpencoder_1/bias*
_output_shapes
:P*
dtype0
|
encoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*!
shared_nameencoder_1/kernel
u
$encoder_1/kernel/Read/ReadVariableOpReadVariableOpencoder_1/kernel*
_output_shapes

:PP*
dtype0
r
elnorm_0/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameelnorm_0/beta
k
!elnorm_0/beta/Read/ReadVariableOpReadVariableOpelnorm_0/beta*
_output_shapes
:P*
dtype0
t
elnorm_0/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameelnorm_0/gamma
m
"elnorm_0/gamma/Read/ReadVariableOpReadVariableOpelnorm_0/gamma*
_output_shapes
:P*
dtype0
�
ebatch_norm_0/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*.
shared_nameebatch_norm_0/moving_variance
�
1ebatch_norm_0/moving_variance/Read/ReadVariableOpReadVariableOpebatch_norm_0/moving_variance*
_output_shapes
:P*
dtype0
�
ebatch_norm_0/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameebatch_norm_0/moving_mean
�
-ebatch_norm_0/moving_mean/Read/ReadVariableOpReadVariableOpebatch_norm_0/moving_mean*
_output_shapes
:P*
dtype0
|
ebatch_norm_0/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*#
shared_nameebatch_norm_0/beta
u
&ebatch_norm_0/beta/Read/ReadVariableOpReadVariableOpebatch_norm_0/beta*
_output_shapes
:P*
dtype0
~
ebatch_norm_0/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameebatch_norm_0/gamma
w
'ebatch_norm_0/gamma/Read/ReadVariableOpReadVariableOpebatch_norm_0/gamma*
_output_shapes
:P*
dtype0
t
encoder_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameencoder_0/bias
m
"encoder_0/bias/Read/ReadVariableOpReadVariableOpencoder_0/bias*
_output_shapes
:P*
dtype0
|
encoder_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*!
shared_nameencoder_0/kernel
u
$encoder_0/kernel/Read/ReadVariableOpReadVariableOpencoder_0/kernel*
_output_shapes

:P*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
layer_with_weights-12
layer-13
layer-14
layer_with_weights-13
layer-15
layer_with_weights-14
layer-16
layer-17
layer_with_weights-15
layer-18
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+axis
	,gamma
-beta
.moving_mean
/moving_variance*
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6axis
	7gamma
8beta*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias*
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance*
�
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Raxis
	Sgamma
Tbeta*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias*
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance*
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses
naxis
	ogamma
pbeta*
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias*
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
#0
$1
,2
-3
.4
/5
76
87
?8
@9
H10
I11
J12
K13
S14
T15
[16
\17
d18
e19
f20
g21
o22
p23
w24
x25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41*
�
#0
$1
,2
-3
74
85
?6
@7
H8
I9
S10
T11
[12
\13
d14
e15
o16
p17
w18
x19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31*
*
�0
�1
�2
�3
�4* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�beta_1
�beta_2

�decay
�learning_rate
	�iter#m�$m�,m�-m�7m�8m�?m�@m�Hm�Im�Sm�Tm�[m�\m�dm�em�om�pm�wm�xm�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�#v�$v�,v�-v�7v�8v�?v�@v�Hv�Iv�Sv�Tv�[v�\v�dv�ev�ov�pv�wv�xv�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�*

�serving_default* 

#0
$1*

#0
$1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEencoder_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEencoder_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
,0
-1
.2
/3*

,0
-1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
b\
VARIABLE_VALUEebatch_norm_0/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEebatch_norm_0/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEebatch_norm_0/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEebatch_norm_0/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
]W
VARIABLE_VALUEelnorm_0/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEelnorm_0/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*

?0
@1*

?0
@1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEencoder_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEencoder_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
H0
I1
J2
K3*

H0
I1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
b\
VARIABLE_VALUEebatch_norm_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEebatch_norm_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEebatch_norm_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEebatch_norm_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

S0
T1*

S0
T1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
]W
VARIABLE_VALUEelnorm_1/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEelnorm_1/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*

[0
\1*

[0
\1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEencoder_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEencoder_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
d0
e1
f2
g3*

d0
e1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
b\
VARIABLE_VALUEebatch_norm_2/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEebatch_norm_2/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEebatch_norm_2/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEebatch_norm_2/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

o0
p1*

o0
p1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
]W
VARIABLE_VALUEelnorm_2/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEelnorm_2/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*

w0
x1*

w0
x1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEencoder_3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEencoder_3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
c]
VARIABLE_VALUEebatch_norm_3/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEebatch_norm_3/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEebatch_norm_3/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEebatch_norm_3/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
^X
VARIABLE_VALUEelnorm_3/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEelnorm_3/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
c]
VARIABLE_VALUEebottleneck/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEebottleneck/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEregress_1/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEregress_1/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
`Z
VARIABLE_VALUEreg_norm_1/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEreg_norm_1/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEreg_norm_1/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEreg_norm_1/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEdense/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
dense/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
N
.0
/1
J2
K3
f4
g5
�6
�7
�8
�9*
�
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
11
12
13
14
15
16
17
18*

�0*
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
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 


�0* 
* 
* 
* 

.0
/1*
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
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 

J0
K1*
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
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 

f0
g1*
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
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 

�0
�1*
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


�0* 
* 
* 
* 

�0
�1*
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
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/encoder_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/encoder_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEAdam/ebatch_norm_0/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/ebatch_norm_0/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/elnorm_0/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/elnorm_0/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/encoder_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/encoder_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEAdam/ebatch_norm_1/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/ebatch_norm_1/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/elnorm_1/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/elnorm_1/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/encoder_2/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/encoder_2/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEAdam/ebatch_norm_2/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/ebatch_norm_2/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/elnorm_2/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/elnorm_2/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/encoder_3/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/encoder_3/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/ebatch_norm_3/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/ebatch_norm_3/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/elnorm_3/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/elnorm_3/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/ebottleneck/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/ebottleneck/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/regress_1/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/regress_1/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/reg_norm_1/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/reg_norm_1/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/dense/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/encoder_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/encoder_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEAdam/ebatch_norm_0/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/ebatch_norm_0/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/elnorm_0/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/elnorm_0/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/encoder_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/encoder_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEAdam/ebatch_norm_1/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/ebatch_norm_1/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/elnorm_1/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/elnorm_1/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/encoder_2/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/encoder_2/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEAdam/ebatch_norm_2/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/ebatch_norm_2/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/elnorm_2/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/elnorm_2/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/encoder_3/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/encoder_3/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/ebatch_norm_3/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/ebatch_norm_3/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/elnorm_3/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/elnorm_3/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/ebottleneck/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/ebottleneck/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/regress_1/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/regress_1/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/reg_norm_1/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/reg_norm_1/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/dense/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
serving_default_intensityPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�	
StatefulPartitionedCallStatefulPartitionedCallserving_default_intensityencoder_0/kernelencoder_0/biasebatch_norm_0/moving_varianceebatch_norm_0/gammaebatch_norm_0/moving_meanebatch_norm_0/betaelnorm_0/gammaelnorm_0/betaencoder_1/kernelencoder_1/biasebatch_norm_1/moving_varianceebatch_norm_1/gammaebatch_norm_1/moving_meanebatch_norm_1/betaelnorm_1/gammaelnorm_1/betaencoder_2/kernelencoder_2/biasebatch_norm_2/moving_varianceebatch_norm_2/gammaebatch_norm_2/moving_meanebatch_norm_2/betaelnorm_2/gammaelnorm_2/betaencoder_3/kernelencoder_3/biasebatch_norm_3/moving_varianceebatch_norm_3/gammaebatch_norm_3/moving_meanebatch_norm_3/betaelnorm_3/gammaelnorm_3/betaebottleneck/kernelebottleneck/biasregress_1/kernelregress_1/biasreg_norm_1/moving_variancereg_norm_1/gammareg_norm_1/moving_meanreg_norm_1/betadense/kernel
dense/bias*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *.
f)R'
%__inference_signature_wrapper_1359633
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�(
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$encoder_0/kernel/Read/ReadVariableOp"encoder_0/bias/Read/ReadVariableOp'ebatch_norm_0/gamma/Read/ReadVariableOp&ebatch_norm_0/beta/Read/ReadVariableOp-ebatch_norm_0/moving_mean/Read/ReadVariableOp1ebatch_norm_0/moving_variance/Read/ReadVariableOp"elnorm_0/gamma/Read/ReadVariableOp!elnorm_0/beta/Read/ReadVariableOp$encoder_1/kernel/Read/ReadVariableOp"encoder_1/bias/Read/ReadVariableOp'ebatch_norm_1/gamma/Read/ReadVariableOp&ebatch_norm_1/beta/Read/ReadVariableOp-ebatch_norm_1/moving_mean/Read/ReadVariableOp1ebatch_norm_1/moving_variance/Read/ReadVariableOp"elnorm_1/gamma/Read/ReadVariableOp!elnorm_1/beta/Read/ReadVariableOp$encoder_2/kernel/Read/ReadVariableOp"encoder_2/bias/Read/ReadVariableOp'ebatch_norm_2/gamma/Read/ReadVariableOp&ebatch_norm_2/beta/Read/ReadVariableOp-ebatch_norm_2/moving_mean/Read/ReadVariableOp1ebatch_norm_2/moving_variance/Read/ReadVariableOp"elnorm_2/gamma/Read/ReadVariableOp!elnorm_2/beta/Read/ReadVariableOp$encoder_3/kernel/Read/ReadVariableOp"encoder_3/bias/Read/ReadVariableOp'ebatch_norm_3/gamma/Read/ReadVariableOp&ebatch_norm_3/beta/Read/ReadVariableOp-ebatch_norm_3/moving_mean/Read/ReadVariableOp1ebatch_norm_3/moving_variance/Read/ReadVariableOp"elnorm_3/gamma/Read/ReadVariableOp!elnorm_3/beta/Read/ReadVariableOp&ebottleneck/kernel/Read/ReadVariableOp$ebottleneck/bias/Read/ReadVariableOp$regress_1/kernel/Read/ReadVariableOp"regress_1/bias/Read/ReadVariableOp$reg_norm_1/gamma/Read/ReadVariableOp#reg_norm_1/beta/Read/ReadVariableOp*reg_norm_1/moving_mean/Read/ReadVariableOp.reg_norm_1/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/encoder_0/kernel/m/Read/ReadVariableOp)Adam/encoder_0/bias/m/Read/ReadVariableOp.Adam/ebatch_norm_0/gamma/m/Read/ReadVariableOp-Adam/ebatch_norm_0/beta/m/Read/ReadVariableOp)Adam/elnorm_0/gamma/m/Read/ReadVariableOp(Adam/elnorm_0/beta/m/Read/ReadVariableOp+Adam/encoder_1/kernel/m/Read/ReadVariableOp)Adam/encoder_1/bias/m/Read/ReadVariableOp.Adam/ebatch_norm_1/gamma/m/Read/ReadVariableOp-Adam/ebatch_norm_1/beta/m/Read/ReadVariableOp)Adam/elnorm_1/gamma/m/Read/ReadVariableOp(Adam/elnorm_1/beta/m/Read/ReadVariableOp+Adam/encoder_2/kernel/m/Read/ReadVariableOp)Adam/encoder_2/bias/m/Read/ReadVariableOp.Adam/ebatch_norm_2/gamma/m/Read/ReadVariableOp-Adam/ebatch_norm_2/beta/m/Read/ReadVariableOp)Adam/elnorm_2/gamma/m/Read/ReadVariableOp(Adam/elnorm_2/beta/m/Read/ReadVariableOp+Adam/encoder_3/kernel/m/Read/ReadVariableOp)Adam/encoder_3/bias/m/Read/ReadVariableOp.Adam/ebatch_norm_3/gamma/m/Read/ReadVariableOp-Adam/ebatch_norm_3/beta/m/Read/ReadVariableOp)Adam/elnorm_3/gamma/m/Read/ReadVariableOp(Adam/elnorm_3/beta/m/Read/ReadVariableOp-Adam/ebottleneck/kernel/m/Read/ReadVariableOp+Adam/ebottleneck/bias/m/Read/ReadVariableOp+Adam/regress_1/kernel/m/Read/ReadVariableOp)Adam/regress_1/bias/m/Read/ReadVariableOp+Adam/reg_norm_1/gamma/m/Read/ReadVariableOp*Adam/reg_norm_1/beta/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp+Adam/encoder_0/kernel/v/Read/ReadVariableOp)Adam/encoder_0/bias/v/Read/ReadVariableOp.Adam/ebatch_norm_0/gamma/v/Read/ReadVariableOp-Adam/ebatch_norm_0/beta/v/Read/ReadVariableOp)Adam/elnorm_0/gamma/v/Read/ReadVariableOp(Adam/elnorm_0/beta/v/Read/ReadVariableOp+Adam/encoder_1/kernel/v/Read/ReadVariableOp)Adam/encoder_1/bias/v/Read/ReadVariableOp.Adam/ebatch_norm_1/gamma/v/Read/ReadVariableOp-Adam/ebatch_norm_1/beta/v/Read/ReadVariableOp)Adam/elnorm_1/gamma/v/Read/ReadVariableOp(Adam/elnorm_1/beta/v/Read/ReadVariableOp+Adam/encoder_2/kernel/v/Read/ReadVariableOp)Adam/encoder_2/bias/v/Read/ReadVariableOp.Adam/ebatch_norm_2/gamma/v/Read/ReadVariableOp-Adam/ebatch_norm_2/beta/v/Read/ReadVariableOp)Adam/elnorm_2/gamma/v/Read/ReadVariableOp(Adam/elnorm_2/beta/v/Read/ReadVariableOp+Adam/encoder_3/kernel/v/Read/ReadVariableOp)Adam/encoder_3/bias/v/Read/ReadVariableOp.Adam/ebatch_norm_3/gamma/v/Read/ReadVariableOp-Adam/ebatch_norm_3/beta/v/Read/ReadVariableOp)Adam/elnorm_3/gamma/v/Read/ReadVariableOp(Adam/elnorm_3/beta/v/Read/ReadVariableOp-Adam/ebottleneck/kernel/v/Read/ReadVariableOp+Adam/ebottleneck/bias/v/Read/ReadVariableOp+Adam/regress_1/kernel/v/Read/ReadVariableOp)Adam/regress_1/bias/v/Read/ReadVariableOp+Adam/reg_norm_1/gamma/v/Read/ReadVariableOp*Adam/reg_norm_1/beta/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*~
Tinw
u2s	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *)
f$R"
 __inference__traced_save_1361752
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameencoder_0/kernelencoder_0/biasebatch_norm_0/gammaebatch_norm_0/betaebatch_norm_0/moving_meanebatch_norm_0/moving_varianceelnorm_0/gammaelnorm_0/betaencoder_1/kernelencoder_1/biasebatch_norm_1/gammaebatch_norm_1/betaebatch_norm_1/moving_meanebatch_norm_1/moving_varianceelnorm_1/gammaelnorm_1/betaencoder_2/kernelencoder_2/biasebatch_norm_2/gammaebatch_norm_2/betaebatch_norm_2/moving_meanebatch_norm_2/moving_varianceelnorm_2/gammaelnorm_2/betaencoder_3/kernelencoder_3/biasebatch_norm_3/gammaebatch_norm_3/betaebatch_norm_3/moving_meanebatch_norm_3/moving_varianceelnorm_3/gammaelnorm_3/betaebottleneck/kernelebottleneck/biasregress_1/kernelregress_1/biasreg_norm_1/gammareg_norm_1/betareg_norm_1/moving_meanreg_norm_1/moving_variancedense/kernel
dense/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/encoder_0/kernel/mAdam/encoder_0/bias/mAdam/ebatch_norm_0/gamma/mAdam/ebatch_norm_0/beta/mAdam/elnorm_0/gamma/mAdam/elnorm_0/beta/mAdam/encoder_1/kernel/mAdam/encoder_1/bias/mAdam/ebatch_norm_1/gamma/mAdam/ebatch_norm_1/beta/mAdam/elnorm_1/gamma/mAdam/elnorm_1/beta/mAdam/encoder_2/kernel/mAdam/encoder_2/bias/mAdam/ebatch_norm_2/gamma/mAdam/ebatch_norm_2/beta/mAdam/elnorm_2/gamma/mAdam/elnorm_2/beta/mAdam/encoder_3/kernel/mAdam/encoder_3/bias/mAdam/ebatch_norm_3/gamma/mAdam/ebatch_norm_3/beta/mAdam/elnorm_3/gamma/mAdam/elnorm_3/beta/mAdam/ebottleneck/kernel/mAdam/ebottleneck/bias/mAdam/regress_1/kernel/mAdam/regress_1/bias/mAdam/reg_norm_1/gamma/mAdam/reg_norm_1/beta/mAdam/dense/kernel/mAdam/dense/bias/mAdam/encoder_0/kernel/vAdam/encoder_0/bias/vAdam/ebatch_norm_0/gamma/vAdam/ebatch_norm_0/beta/vAdam/elnorm_0/gamma/vAdam/elnorm_0/beta/vAdam/encoder_1/kernel/vAdam/encoder_1/bias/vAdam/ebatch_norm_1/gamma/vAdam/ebatch_norm_1/beta/vAdam/elnorm_1/gamma/vAdam/elnorm_1/beta/vAdam/encoder_2/kernel/vAdam/encoder_2/bias/vAdam/ebatch_norm_2/gamma/vAdam/ebatch_norm_2/beta/vAdam/elnorm_2/gamma/vAdam/elnorm_2/beta/vAdam/encoder_3/kernel/vAdam/encoder_3/bias/vAdam/ebatch_norm_3/gamma/vAdam/ebatch_norm_3/beta/vAdam/elnorm_3/gamma/vAdam/elnorm_3/beta/vAdam/ebottleneck/kernel/vAdam/ebottleneck/bias/vAdam/regress_1/kernel/vAdam/regress_1/bias/vAdam/reg_norm_1/gamma/vAdam/reg_norm_1/beta/vAdam/dense/kernel/vAdam/dense/bias/v*}
Tinv
t2r*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *,
f'R%
#__inference__traced_restore_1362101��
�

�
H__inference_ebottleneck_layer_call_and_return_conditional_losses_1358513

inputs0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�	
)__inference_model_1_layer_call_fn_1359752

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P

unknown_10:P

unknown_11:P

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:PP

unknown_16:P

unknown_17:P

unknown_18:P

unknown_19:P

unknown_20:P

unknown_21:P

unknown_22:P

unknown_23:PP

unknown_24:P

unknown_25:P

unknown_26:P

unknown_27:P

unknown_28:P

unknown_29:P

unknown_30:P

unknown_31:P

unknown_32:

unknown_33:2

unknown_34:2

unknown_35:2

unknown_36:2

unknown_37:2

unknown_38:2

unknown_39:2

unknown_40:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1358613o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_regress_1_layer_call_fn_1361200

inputs
unknown:2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_regress_1_layer_call_and_return_conditional_losses_1358544o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_elnorm_0_layer_call_and_return_conditional_losses_1360692

inputs+
mul_2_readvariableop_resource:P)
add_readvariableop_resource:P
identity��add/ReadVariableOp�mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������PJ
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:���������Pn
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:P*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:P*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PV
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������Pr
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1357793

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Pz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
/__inference_ebatch_norm_3_layer_call_fn_1361042

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1358039o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
E
)__inference_dropout_layer_call_fn_1361307

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1358722`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1360763

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Pz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
/__inference_ebatch_norm_2_layer_call_fn_1360899

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1358004o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�E
#__inference__traced_restore_1362101
file_prefix3
!assignvariableop_encoder_0_kernel:P/
!assignvariableop_1_encoder_0_bias:P4
&assignvariableop_2_ebatch_norm_0_gamma:P3
%assignvariableop_3_ebatch_norm_0_beta:P:
,assignvariableop_4_ebatch_norm_0_moving_mean:P>
0assignvariableop_5_ebatch_norm_0_moving_variance:P/
!assignvariableop_6_elnorm_0_gamma:P.
 assignvariableop_7_elnorm_0_beta:P5
#assignvariableop_8_encoder_1_kernel:PP/
!assignvariableop_9_encoder_1_bias:P5
'assignvariableop_10_ebatch_norm_1_gamma:P4
&assignvariableop_11_ebatch_norm_1_beta:P;
-assignvariableop_12_ebatch_norm_1_moving_mean:P?
1assignvariableop_13_ebatch_norm_1_moving_variance:P0
"assignvariableop_14_elnorm_1_gamma:P/
!assignvariableop_15_elnorm_1_beta:P6
$assignvariableop_16_encoder_2_kernel:PP0
"assignvariableop_17_encoder_2_bias:P5
'assignvariableop_18_ebatch_norm_2_gamma:P4
&assignvariableop_19_ebatch_norm_2_beta:P;
-assignvariableop_20_ebatch_norm_2_moving_mean:P?
1assignvariableop_21_ebatch_norm_2_moving_variance:P0
"assignvariableop_22_elnorm_2_gamma:P/
!assignvariableop_23_elnorm_2_beta:P6
$assignvariableop_24_encoder_3_kernel:PP0
"assignvariableop_25_encoder_3_bias:P5
'assignvariableop_26_ebatch_norm_3_gamma:P4
&assignvariableop_27_ebatch_norm_3_beta:P;
-assignvariableop_28_ebatch_norm_3_moving_mean:P?
1assignvariableop_29_ebatch_norm_3_moving_variance:P0
"assignvariableop_30_elnorm_3_gamma:P/
!assignvariableop_31_elnorm_3_beta:P8
&assignvariableop_32_ebottleneck_kernel:P2
$assignvariableop_33_ebottleneck_bias:6
$assignvariableop_34_regress_1_kernel:20
"assignvariableop_35_regress_1_bias:22
$assignvariableop_36_reg_norm_1_gamma:21
#assignvariableop_37_reg_norm_1_beta:28
*assignvariableop_38_reg_norm_1_moving_mean:2<
.assignvariableop_39_reg_norm_1_moving_variance:22
 assignvariableop_40_dense_kernel:2,
assignvariableop_41_dense_bias:$
assignvariableop_42_beta_1: $
assignvariableop_43_beta_2: #
assignvariableop_44_decay: +
!assignvariableop_45_learning_rate: '
assignvariableop_46_adam_iter:	 #
assignvariableop_47_total: #
assignvariableop_48_count: =
+assignvariableop_49_adam_encoder_0_kernel_m:P7
)assignvariableop_50_adam_encoder_0_bias_m:P<
.assignvariableop_51_adam_ebatch_norm_0_gamma_m:P;
-assignvariableop_52_adam_ebatch_norm_0_beta_m:P7
)assignvariableop_53_adam_elnorm_0_gamma_m:P6
(assignvariableop_54_adam_elnorm_0_beta_m:P=
+assignvariableop_55_adam_encoder_1_kernel_m:PP7
)assignvariableop_56_adam_encoder_1_bias_m:P<
.assignvariableop_57_adam_ebatch_norm_1_gamma_m:P;
-assignvariableop_58_adam_ebatch_norm_1_beta_m:P7
)assignvariableop_59_adam_elnorm_1_gamma_m:P6
(assignvariableop_60_adam_elnorm_1_beta_m:P=
+assignvariableop_61_adam_encoder_2_kernel_m:PP7
)assignvariableop_62_adam_encoder_2_bias_m:P<
.assignvariableop_63_adam_ebatch_norm_2_gamma_m:P;
-assignvariableop_64_adam_ebatch_norm_2_beta_m:P7
)assignvariableop_65_adam_elnorm_2_gamma_m:P6
(assignvariableop_66_adam_elnorm_2_beta_m:P=
+assignvariableop_67_adam_encoder_3_kernel_m:PP7
)assignvariableop_68_adam_encoder_3_bias_m:P<
.assignvariableop_69_adam_ebatch_norm_3_gamma_m:P;
-assignvariableop_70_adam_ebatch_norm_3_beta_m:P7
)assignvariableop_71_adam_elnorm_3_gamma_m:P6
(assignvariableop_72_adam_elnorm_3_beta_m:P?
-assignvariableop_73_adam_ebottleneck_kernel_m:P9
+assignvariableop_74_adam_ebottleneck_bias_m:=
+assignvariableop_75_adam_regress_1_kernel_m:27
)assignvariableop_76_adam_regress_1_bias_m:29
+assignvariableop_77_adam_reg_norm_1_gamma_m:28
*assignvariableop_78_adam_reg_norm_1_beta_m:29
'assignvariableop_79_adam_dense_kernel_m:23
%assignvariableop_80_adam_dense_bias_m:=
+assignvariableop_81_adam_encoder_0_kernel_v:P7
)assignvariableop_82_adam_encoder_0_bias_v:P<
.assignvariableop_83_adam_ebatch_norm_0_gamma_v:P;
-assignvariableop_84_adam_ebatch_norm_0_beta_v:P7
)assignvariableop_85_adam_elnorm_0_gamma_v:P6
(assignvariableop_86_adam_elnorm_0_beta_v:P=
+assignvariableop_87_adam_encoder_1_kernel_v:PP7
)assignvariableop_88_adam_encoder_1_bias_v:P<
.assignvariableop_89_adam_ebatch_norm_1_gamma_v:P;
-assignvariableop_90_adam_ebatch_norm_1_beta_v:P7
)assignvariableop_91_adam_elnorm_1_gamma_v:P6
(assignvariableop_92_adam_elnorm_1_beta_v:P=
+assignvariableop_93_adam_encoder_2_kernel_v:PP7
)assignvariableop_94_adam_encoder_2_bias_v:P<
.assignvariableop_95_adam_ebatch_norm_2_gamma_v:P;
-assignvariableop_96_adam_ebatch_norm_2_beta_v:P7
)assignvariableop_97_adam_elnorm_2_gamma_v:P6
(assignvariableop_98_adam_elnorm_2_beta_v:P=
+assignvariableop_99_adam_encoder_3_kernel_v:PP8
*assignvariableop_100_adam_encoder_3_bias_v:P=
/assignvariableop_101_adam_ebatch_norm_3_gamma_v:P<
.assignvariableop_102_adam_ebatch_norm_3_beta_v:P8
*assignvariableop_103_adam_elnorm_3_gamma_v:P7
)assignvariableop_104_adam_elnorm_3_beta_v:P@
.assignvariableop_105_adam_ebottleneck_kernel_v:P:
,assignvariableop_106_adam_ebottleneck_bias_v:>
,assignvariableop_107_adam_regress_1_kernel_v:28
*assignvariableop_108_adam_regress_1_bias_v:2:
,assignvariableop_109_adam_reg_norm_1_gamma_v:29
+assignvariableop_110_adam_reg_norm_1_beta_v:2:
(assignvariableop_111_adam_dense_kernel_v:24
&assignvariableop_112_adam_dense_bias_v:
identity_114��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�@
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:r*
dtype0*�?
value�?B�?rB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:r*
dtype0*�
value�B�rB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypesv
t2r	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_encoder_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_encoder_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp&assignvariableop_2_ebatch_norm_0_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp%assignvariableop_3_ebatch_norm_0_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_ebatch_norm_0_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp0assignvariableop_5_ebatch_norm_0_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_elnorm_0_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_elnorm_0_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_encoder_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_encoder_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp'assignvariableop_10_ebatch_norm_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp&assignvariableop_11_ebatch_norm_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp-assignvariableop_12_ebatch_norm_1_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp1assignvariableop_13_ebatch_norm_1_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_elnorm_1_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_elnorm_1_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp$assignvariableop_16_encoder_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp"assignvariableop_17_encoder_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_ebatch_norm_2_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp&assignvariableop_19_ebatch_norm_2_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp-assignvariableop_20_ebatch_norm_2_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp1assignvariableop_21_ebatch_norm_2_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp"assignvariableop_22_elnorm_2_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_elnorm_2_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp$assignvariableop_24_encoder_3_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_encoder_3_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_ebatch_norm_3_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp&assignvariableop_27_ebatch_norm_3_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp-assignvariableop_28_ebatch_norm_3_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp1assignvariableop_29_ebatch_norm_3_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp"assignvariableop_30_elnorm_3_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp!assignvariableop_31_elnorm_3_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp&assignvariableop_32_ebottleneck_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp$assignvariableop_33_ebottleneck_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp$assignvariableop_34_regress_1_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp"assignvariableop_35_regress_1_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp$assignvariableop_36_reg_norm_1_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp#assignvariableop_37_reg_norm_1_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp*assignvariableop_38_reg_norm_1_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp.assignvariableop_39_reg_norm_1_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp assignvariableop_40_dense_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpassignvariableop_41_dense_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpassignvariableop_42_beta_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpassignvariableop_43_beta_2Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpassignvariableop_44_decayIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp!assignvariableop_45_learning_rateIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpassignvariableop_46_adam_iterIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpassignvariableop_47_totalIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpassignvariableop_48_countIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_encoder_0_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_encoder_0_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp.assignvariableop_51_adam_ebatch_norm_0_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp-assignvariableop_52_adam_ebatch_norm_0_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_elnorm_0_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_elnorm_0_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_encoder_1_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_encoder_1_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp.assignvariableop_57_adam_ebatch_norm_1_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp-assignvariableop_58_adam_ebatch_norm_1_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_elnorm_1_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_elnorm_1_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_encoder_2_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_encoder_2_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp.assignvariableop_63_adam_ebatch_norm_2_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp-assignvariableop_64_adam_ebatch_norm_2_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp)assignvariableop_65_adam_elnorm_2_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_elnorm_2_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_encoder_3_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_encoder_3_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp.assignvariableop_69_adam_ebatch_norm_3_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp-assignvariableop_70_adam_ebatch_norm_3_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp)assignvariableop_71_adam_elnorm_3_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_elnorm_3_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp-assignvariableop_73_adam_ebottleneck_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp+assignvariableop_74_adam_ebottleneck_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_regress_1_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_regress_1_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_reg_norm_1_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp*assignvariableop_78_adam_reg_norm_1_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp'assignvariableop_79_adam_dense_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp%assignvariableop_80_adam_dense_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_encoder_0_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_encoder_0_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp.assignvariableop_83_adam_ebatch_norm_0_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp-assignvariableop_84_adam_ebatch_norm_0_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp)assignvariableop_85_adam_elnorm_0_gamma_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_elnorm_0_beta_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp+assignvariableop_87_adam_encoder_1_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp)assignvariableop_88_adam_encoder_1_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp.assignvariableop_89_adam_ebatch_norm_1_gamma_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp-assignvariableop_90_adam_ebatch_norm_1_beta_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp)assignvariableop_91_adam_elnorm_1_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_elnorm_1_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_encoder_2_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_encoder_2_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp.assignvariableop_95_adam_ebatch_norm_2_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp-assignvariableop_96_adam_ebatch_norm_2_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp)assignvariableop_97_adam_elnorm_2_gamma_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_elnorm_2_beta_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp+assignvariableop_99_adam_encoder_3_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp*assignvariableop_100_adam_encoder_3_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp/assignvariableop_101_adam_ebatch_norm_3_gamma_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp.assignvariableop_102_adam_ebatch_norm_3_beta_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp*assignvariableop_103_adam_elnorm_3_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_elnorm_3_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp.assignvariableop_105_adam_ebottleneck_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp,assignvariableop_106_adam_ebottleneck_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOp,assignvariableop_107_adam_regress_1_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp*assignvariableop_108_adam_regress_1_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_reg_norm_1_gamma_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp+assignvariableop_110_adam_reg_norm_1_beta_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp(assignvariableop_111_adam_dense_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp&assignvariableop_112_adam_dense_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_113Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_114IdentityIdentity_113:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_114Identity_114:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122*
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
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_1361312

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�	
�
B__inference_dense_layer_call_and_return_conditional_losses_1358576

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
/__inference_ebatch_norm_1_layer_call_fn_1360743

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1357922o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
F__inference_encoder_0_layer_call_and_return_conditional_losses_1360561

inputs0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2encoder_0/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
2encoder_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0�
#encoder_0/kernel/Regularizer/SquareSquare:encoder_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Ps
"encoder_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_0/kernel/Regularizer/SumSum'encoder_0/kernel/Regularizer/Square:y:0+encoder_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_0/kernel/Regularizer/mulMul+encoder_0/kernel/Regularizer/mul/x:output:0)encoder_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^encoder_0/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2encoder_0/kernel/Regularizer/Square/ReadVariableOp2encoder_0/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_encoder_3_layer_call_and_return_conditional_losses_1358439

inputs0
matmul_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2encoder_3/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
2encoder_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_3/kernel/Regularizer/SquareSquare:encoder_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_3/kernel/Regularizer/SumSum'encoder_3/kernel/Regularizer/Square:y:0+encoder_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_3/kernel/Regularizer/mulMul+encoder_3/kernel/Regularizer/mul/x:output:0)encoder_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^encoder_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2encoder_3/kernel/Regularizer/Square/ReadVariableOp2encoder_3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�	
)__inference_model_1_layer_call_fn_1358700
	intensity
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P

unknown_10:P

unknown_11:P

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:PP

unknown_16:P

unknown_17:P

unknown_18:P

unknown_19:P

unknown_20:P

unknown_21:P

unknown_22:P

unknown_23:PP

unknown_24:P

unknown_25:P

unknown_26:P

unknown_27:P

unknown_28:P

unknown_29:P

unknown_30:P

unknown_31:P

unknown_32:

unknown_33:2

unknown_34:2

unknown_35:2

unknown_36:2

unknown_37:2

unknown_38:2

unknown_39:2

unknown_40:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	intensityunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1358613o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:���������
#
_user_specified_name	intensity
�%
�
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1361297

inputs5
'assignmovingavg_readvariableop_resource:27
)assignmovingavg_1_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:2/
!batchnorm_readvariableop_resource:2
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:2*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:2*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1360607

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Pz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1358121

inputs/
!batchnorm_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:21
#batchnorm_readvariableop_1_resource:21
#batchnorm_readvariableop_2_resource:2
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
__inference_loss_fn_4_1361390M
;regress_1_kernel_regularizer_square_readvariableop_resource:2
identity��2regress_1/kernel/Regularizer/Square/ReadVariableOp�
2regress_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;regress_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:2*
dtype0�
#regress_1/kernel/Regularizer/SquareSquare:regress_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2s
"regress_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 regress_1/kernel/Regularizer/SumSum'regress_1/kernel/Regularizer/Square:y:0+regress_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"regress_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 regress_1/kernel/Regularizer/mulMul+regress_1/kernel/Regularizer/mul/x:output:0)regress_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$regress_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^regress_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2regress_1/kernel/Regularizer/Square/ReadVariableOp2regress_1/kernel/Regularizer/Square/ReadVariableOp
�
�	
%__inference_signature_wrapper_1359633
	intensity
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P

unknown_10:P

unknown_11:P

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:PP

unknown_16:P

unknown_17:P

unknown_18:P

unknown_19:P

unknown_20:P

unknown_21:P

unknown_22:P

unknown_23:PP

unknown_24:P

unknown_25:P

unknown_26:P

unknown_27:P

unknown_28:P

unknown_29:P

unknown_30:P

unknown_31:P

unknown_32:

unknown_33:2

unknown_34:2

unknown_35:2

unknown_36:2

unknown_37:2

unknown_38:2

unknown_39:2

unknown_40:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	intensityunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *+
f&R$
"__inference__wrapped_model_1357769o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:���������
#
_user_specified_name	intensity
�%
�
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1360953

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Ph
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1358039

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Pz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�
D__inference_model_1_layer_call_and_return_conditional_losses_1359058

inputs#
encoder_0_1358925:P
encoder_0_1358927:P#
ebatch_norm_0_1358930:P#
ebatch_norm_0_1358932:P#
ebatch_norm_0_1358934:P#
ebatch_norm_0_1358936:P
elnorm_0_1358939:P
elnorm_0_1358941:P#
encoder_1_1358944:PP
encoder_1_1358946:P#
ebatch_norm_1_1358949:P#
ebatch_norm_1_1358951:P#
ebatch_norm_1_1358953:P#
ebatch_norm_1_1358955:P
elnorm_1_1358958:P
elnorm_1_1358960:P#
encoder_2_1358963:PP
encoder_2_1358965:P#
ebatch_norm_2_1358968:P#
ebatch_norm_2_1358970:P#
ebatch_norm_2_1358972:P#
ebatch_norm_2_1358974:P
elnorm_2_1358977:P
elnorm_2_1358979:P#
encoder_3_1358982:PP
encoder_3_1358984:P#
ebatch_norm_3_1358987:P#
ebatch_norm_3_1358989:P#
ebatch_norm_3_1358991:P#
ebatch_norm_3_1358993:P
elnorm_3_1358996:P
elnorm_3_1358998:P%
ebottleneck_1359001:P!
ebottleneck_1359003:#
regress_1_1359007:2
regress_1_1359009:2 
reg_norm_1_1359012:2 
reg_norm_1_1359014:2 
reg_norm_1_1359016:2 
reg_norm_1_1359018:2
dense_1359022:2
dense_1359024:
identity��dense/StatefulPartitionedCall�%ebatch_norm_0/StatefulPartitionedCall�%ebatch_norm_1/StatefulPartitionedCall�%ebatch_norm_2/StatefulPartitionedCall�%ebatch_norm_3/StatefulPartitionedCall�#ebottleneck/StatefulPartitionedCall� elnorm_0/StatefulPartitionedCall� elnorm_1/StatefulPartitionedCall� elnorm_2/StatefulPartitionedCall� elnorm_3/StatefulPartitionedCall�!encoder_0/StatefulPartitionedCall�2encoder_0/kernel/Regularizer/Square/ReadVariableOp�!encoder_1/StatefulPartitionedCall�2encoder_1/kernel/Regularizer/Square/ReadVariableOp�!encoder_2/StatefulPartitionedCall�2encoder_2/kernel/Regularizer/Square/ReadVariableOp�!encoder_3/StatefulPartitionedCall�2encoder_3/kernel/Regularizer/Square/ReadVariableOp�"reg_norm_1/StatefulPartitionedCall�!regress_1/StatefulPartitionedCall�2regress_1/kernel/Regularizer/Square/ReadVariableOp�
!encoder_0/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_0_1358925encoder_0_1358927*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_0_layer_call_and_return_conditional_losses_1358202�
%ebatch_norm_0/StatefulPartitionedCallStatefulPartitionedCall*encoder_0/StatefulPartitionedCall:output:0ebatch_norm_0_1358930ebatch_norm_0_1358932ebatch_norm_0_1358934ebatch_norm_0_1358936*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1357840�
 elnorm_0/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_0/StatefulPartitionedCall:output:0elnorm_0_1358939elnorm_0_1358941*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_0_layer_call_and_return_conditional_losses_1358259�
!encoder_1/StatefulPartitionedCallStatefulPartitionedCall)elnorm_0/StatefulPartitionedCall:output:0encoder_1_1358944encoder_1_1358946*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_1_layer_call_and_return_conditional_losses_1358281�
%ebatch_norm_1/StatefulPartitionedCallStatefulPartitionedCall*encoder_1/StatefulPartitionedCall:output:0ebatch_norm_1_1358949ebatch_norm_1_1358951ebatch_norm_1_1358953ebatch_norm_1_1358955*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1357922�
 elnorm_1/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_1/StatefulPartitionedCall:output:0elnorm_1_1358958elnorm_1_1358960*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_1_layer_call_and_return_conditional_losses_1358338�
!encoder_2/StatefulPartitionedCallStatefulPartitionedCall)elnorm_1/StatefulPartitionedCall:output:0encoder_2_1358963encoder_2_1358965*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_2_layer_call_and_return_conditional_losses_1358360�
%ebatch_norm_2/StatefulPartitionedCallStatefulPartitionedCall*encoder_2/StatefulPartitionedCall:output:0ebatch_norm_2_1358968ebatch_norm_2_1358970ebatch_norm_2_1358972ebatch_norm_2_1358974*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1358004�
 elnorm_2/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_2/StatefulPartitionedCall:output:0elnorm_2_1358977elnorm_2_1358979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_2_layer_call_and_return_conditional_losses_1358417�
!encoder_3/StatefulPartitionedCallStatefulPartitionedCall)elnorm_2/StatefulPartitionedCall:output:0encoder_3_1358982encoder_3_1358984*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_3_layer_call_and_return_conditional_losses_1358439�
%ebatch_norm_3/StatefulPartitionedCallStatefulPartitionedCall*encoder_3/StatefulPartitionedCall:output:0ebatch_norm_3_1358987ebatch_norm_3_1358989ebatch_norm_3_1358991ebatch_norm_3_1358993*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1358086�
 elnorm_3/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_3/StatefulPartitionedCall:output:0elnorm_3_1358996elnorm_3_1358998*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_3_layer_call_and_return_conditional_losses_1358496�
#ebottleneck/StatefulPartitionedCallStatefulPartitionedCall)elnorm_3/StatefulPartitionedCall:output:0ebottleneck_1359001ebottleneck_1359003*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_ebottleneck_layer_call_and_return_conditional_losses_1358513�
flatten/PartitionedCallPartitionedCall,ebottleneck/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1358525�
!regress_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0regress_1_1359007regress_1_1359009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_regress_1_layer_call_and_return_conditional_losses_1358544�
"reg_norm_1/StatefulPartitionedCallStatefulPartitionedCall*regress_1/StatefulPartitionedCall:output:0reg_norm_1_1359012reg_norm_1_1359014reg_norm_1_1359016reg_norm_1_1359018*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1358168�
dropout/PartitionedCallPartitionedCall+reg_norm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1358722�
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1359022dense_1359024*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1358576�
2encoder_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_0_1358925*
_output_shapes

:P*
dtype0�
#encoder_0/kernel/Regularizer/SquareSquare:encoder_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Ps
"encoder_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_0/kernel/Regularizer/SumSum'encoder_0/kernel/Regularizer/Square:y:0+encoder_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_0/kernel/Regularizer/mulMul+encoder_0/kernel/Regularizer/mul/x:output:0)encoder_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_1_1358944*
_output_shapes

:PP*
dtype0�
#encoder_1/kernel/Regularizer/SquareSquare:encoder_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_1/kernel/Regularizer/SumSum'encoder_1/kernel/Regularizer/Square:y:0+encoder_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_1/kernel/Regularizer/mulMul+encoder_1/kernel/Regularizer/mul/x:output:0)encoder_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_2_1358963*
_output_shapes

:PP*
dtype0�
#encoder_2/kernel/Regularizer/SquareSquare:encoder_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_2/kernel/Regularizer/SumSum'encoder_2/kernel/Regularizer/Square:y:0+encoder_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_2/kernel/Regularizer/mulMul+encoder_2/kernel/Regularizer/mul/x:output:0)encoder_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_3_1358982*
_output_shapes

:PP*
dtype0�
#encoder_3/kernel/Regularizer/SquareSquare:encoder_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_3/kernel/Regularizer/SumSum'encoder_3/kernel/Regularizer/Square:y:0+encoder_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_3/kernel/Regularizer/mulMul+encoder_3/kernel/Regularizer/mul/x:output:0)encoder_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2regress_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpregress_1_1359007*
_output_shapes

:2*
dtype0�
#regress_1/kernel/Regularizer/SquareSquare:regress_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2s
"regress_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 regress_1/kernel/Regularizer/SumSum'regress_1/kernel/Regularizer/Square:y:0+regress_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"regress_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 regress_1/kernel/Regularizer/mulMul+regress_1/kernel/Regularizer/mul/x:output:0)regress_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall&^ebatch_norm_0/StatefulPartitionedCall&^ebatch_norm_1/StatefulPartitionedCall&^ebatch_norm_2/StatefulPartitionedCall&^ebatch_norm_3/StatefulPartitionedCall$^ebottleneck/StatefulPartitionedCall!^elnorm_0/StatefulPartitionedCall!^elnorm_1/StatefulPartitionedCall!^elnorm_2/StatefulPartitionedCall!^elnorm_3/StatefulPartitionedCall"^encoder_0/StatefulPartitionedCall3^encoder_0/kernel/Regularizer/Square/ReadVariableOp"^encoder_1/StatefulPartitionedCall3^encoder_1/kernel/Regularizer/Square/ReadVariableOp"^encoder_2/StatefulPartitionedCall3^encoder_2/kernel/Regularizer/Square/ReadVariableOp"^encoder_3/StatefulPartitionedCall3^encoder_3/kernel/Regularizer/Square/ReadVariableOp#^reg_norm_1/StatefulPartitionedCall"^regress_1/StatefulPartitionedCall3^regress_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%ebatch_norm_0/StatefulPartitionedCall%ebatch_norm_0/StatefulPartitionedCall2N
%ebatch_norm_1/StatefulPartitionedCall%ebatch_norm_1/StatefulPartitionedCall2N
%ebatch_norm_2/StatefulPartitionedCall%ebatch_norm_2/StatefulPartitionedCall2N
%ebatch_norm_3/StatefulPartitionedCall%ebatch_norm_3/StatefulPartitionedCall2J
#ebottleneck/StatefulPartitionedCall#ebottleneck/StatefulPartitionedCall2D
 elnorm_0/StatefulPartitionedCall elnorm_0/StatefulPartitionedCall2D
 elnorm_1/StatefulPartitionedCall elnorm_1/StatefulPartitionedCall2D
 elnorm_2/StatefulPartitionedCall elnorm_2/StatefulPartitionedCall2D
 elnorm_3/StatefulPartitionedCall elnorm_3/StatefulPartitionedCall2F
!encoder_0/StatefulPartitionedCall!encoder_0/StatefulPartitionedCall2h
2encoder_0/kernel/Regularizer/Square/ReadVariableOp2encoder_0/kernel/Regularizer/Square/ReadVariableOp2F
!encoder_1/StatefulPartitionedCall!encoder_1/StatefulPartitionedCall2h
2encoder_1/kernel/Regularizer/Square/ReadVariableOp2encoder_1/kernel/Regularizer/Square/ReadVariableOp2F
!encoder_2/StatefulPartitionedCall!encoder_2/StatefulPartitionedCall2h
2encoder_2/kernel/Regularizer/Square/ReadVariableOp2encoder_2/kernel/Regularizer/Square/ReadVariableOp2F
!encoder_3/StatefulPartitionedCall!encoder_3/StatefulPartitionedCall2h
2encoder_3/kernel/Regularizer/Square/ReadVariableOp2encoder_3/kernel/Regularizer/Square/ReadVariableOp2H
"reg_norm_1/StatefulPartitionedCall"reg_norm_1/StatefulPartitionedCall2F
!regress_1/StatefulPartitionedCall!regress_1/StatefulPartitionedCall2h
2regress_1/kernel/Regularizer/Square/ReadVariableOp2regress_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_encoder_1_layer_call_fn_1360701

inputs
unknown:PP
	unknown_0:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_1_layer_call_and_return_conditional_losses_1358281o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�%
"__inference__wrapped_model_1357769
	intensityB
0model_1_encoder_0_matmul_readvariableop_resource:P?
1model_1_encoder_0_biasadd_readvariableop_resource:PE
7model_1_ebatch_norm_0_batchnorm_readvariableop_resource:PI
;model_1_ebatch_norm_0_batchnorm_mul_readvariableop_resource:PG
9model_1_ebatch_norm_0_batchnorm_readvariableop_1_resource:PG
9model_1_ebatch_norm_0_batchnorm_readvariableop_2_resource:P<
.model_1_elnorm_0_mul_2_readvariableop_resource:P:
,model_1_elnorm_0_add_readvariableop_resource:PB
0model_1_encoder_1_matmul_readvariableop_resource:PP?
1model_1_encoder_1_biasadd_readvariableop_resource:PE
7model_1_ebatch_norm_1_batchnorm_readvariableop_resource:PI
;model_1_ebatch_norm_1_batchnorm_mul_readvariableop_resource:PG
9model_1_ebatch_norm_1_batchnorm_readvariableop_1_resource:PG
9model_1_ebatch_norm_1_batchnorm_readvariableop_2_resource:P<
.model_1_elnorm_1_mul_2_readvariableop_resource:P:
,model_1_elnorm_1_add_readvariableop_resource:PB
0model_1_encoder_2_matmul_readvariableop_resource:PP?
1model_1_encoder_2_biasadd_readvariableop_resource:PE
7model_1_ebatch_norm_2_batchnorm_readvariableop_resource:PI
;model_1_ebatch_norm_2_batchnorm_mul_readvariableop_resource:PG
9model_1_ebatch_norm_2_batchnorm_readvariableop_1_resource:PG
9model_1_ebatch_norm_2_batchnorm_readvariableop_2_resource:P<
.model_1_elnorm_2_mul_2_readvariableop_resource:P:
,model_1_elnorm_2_add_readvariableop_resource:PB
0model_1_encoder_3_matmul_readvariableop_resource:PP?
1model_1_encoder_3_biasadd_readvariableop_resource:PE
7model_1_ebatch_norm_3_batchnorm_readvariableop_resource:PI
;model_1_ebatch_norm_3_batchnorm_mul_readvariableop_resource:PG
9model_1_ebatch_norm_3_batchnorm_readvariableop_1_resource:PG
9model_1_ebatch_norm_3_batchnorm_readvariableop_2_resource:P<
.model_1_elnorm_3_mul_2_readvariableop_resource:P:
,model_1_elnorm_3_add_readvariableop_resource:PD
2model_1_ebottleneck_matmul_readvariableop_resource:PA
3model_1_ebottleneck_biasadd_readvariableop_resource:B
0model_1_regress_1_matmul_readvariableop_resource:2?
1model_1_regress_1_biasadd_readvariableop_resource:2B
4model_1_reg_norm_1_batchnorm_readvariableop_resource:2F
8model_1_reg_norm_1_batchnorm_mul_readvariableop_resource:2D
6model_1_reg_norm_1_batchnorm_readvariableop_1_resource:2D
6model_1_reg_norm_1_batchnorm_readvariableop_2_resource:2>
,model_1_dense_matmul_readvariableop_resource:2;
-model_1_dense_biasadd_readvariableop_resource:
identity��$model_1/dense/BiasAdd/ReadVariableOp�#model_1/dense/MatMul/ReadVariableOp�.model_1/ebatch_norm_0/batchnorm/ReadVariableOp�0model_1/ebatch_norm_0/batchnorm/ReadVariableOp_1�0model_1/ebatch_norm_0/batchnorm/ReadVariableOp_2�2model_1/ebatch_norm_0/batchnorm/mul/ReadVariableOp�.model_1/ebatch_norm_1/batchnorm/ReadVariableOp�0model_1/ebatch_norm_1/batchnorm/ReadVariableOp_1�0model_1/ebatch_norm_1/batchnorm/ReadVariableOp_2�2model_1/ebatch_norm_1/batchnorm/mul/ReadVariableOp�.model_1/ebatch_norm_2/batchnorm/ReadVariableOp�0model_1/ebatch_norm_2/batchnorm/ReadVariableOp_1�0model_1/ebatch_norm_2/batchnorm/ReadVariableOp_2�2model_1/ebatch_norm_2/batchnorm/mul/ReadVariableOp�.model_1/ebatch_norm_3/batchnorm/ReadVariableOp�0model_1/ebatch_norm_3/batchnorm/ReadVariableOp_1�0model_1/ebatch_norm_3/batchnorm/ReadVariableOp_2�2model_1/ebatch_norm_3/batchnorm/mul/ReadVariableOp�*model_1/ebottleneck/BiasAdd/ReadVariableOp�)model_1/ebottleneck/MatMul/ReadVariableOp�#model_1/elnorm_0/add/ReadVariableOp�%model_1/elnorm_0/mul_2/ReadVariableOp�#model_1/elnorm_1/add/ReadVariableOp�%model_1/elnorm_1/mul_2/ReadVariableOp�#model_1/elnorm_2/add/ReadVariableOp�%model_1/elnorm_2/mul_2/ReadVariableOp�#model_1/elnorm_3/add/ReadVariableOp�%model_1/elnorm_3/mul_2/ReadVariableOp�(model_1/encoder_0/BiasAdd/ReadVariableOp�'model_1/encoder_0/MatMul/ReadVariableOp�(model_1/encoder_1/BiasAdd/ReadVariableOp�'model_1/encoder_1/MatMul/ReadVariableOp�(model_1/encoder_2/BiasAdd/ReadVariableOp�'model_1/encoder_2/MatMul/ReadVariableOp�(model_1/encoder_3/BiasAdd/ReadVariableOp�'model_1/encoder_3/MatMul/ReadVariableOp�+model_1/reg_norm_1/batchnorm/ReadVariableOp�-model_1/reg_norm_1/batchnorm/ReadVariableOp_1�-model_1/reg_norm_1/batchnorm/ReadVariableOp_2�/model_1/reg_norm_1/batchnorm/mul/ReadVariableOp�(model_1/regress_1/BiasAdd/ReadVariableOp�'model_1/regress_1/MatMul/ReadVariableOp�
'model_1/encoder_0/MatMul/ReadVariableOpReadVariableOp0model_1_encoder_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0�
model_1/encoder_0/MatMulMatMul	intensity/model_1/encoder_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
(model_1/encoder_0/BiasAdd/ReadVariableOpReadVariableOp1model_1_encoder_0_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_1/encoder_0/BiasAddBiasAdd"model_1/encoder_0/MatMul:product:00model_1/encoder_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
.model_1/ebatch_norm_0/batchnorm/ReadVariableOpReadVariableOp7model_1_ebatch_norm_0_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0j
%model_1/ebatch_norm_0/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#model_1/ebatch_norm_0/batchnorm/addAddV26model_1/ebatch_norm_0/batchnorm/ReadVariableOp:value:0.model_1/ebatch_norm_0/batchnorm/add/y:output:0*
T0*
_output_shapes
:P|
%model_1/ebatch_norm_0/batchnorm/RsqrtRsqrt'model_1/ebatch_norm_0/batchnorm/add:z:0*
T0*
_output_shapes
:P�
2model_1/ebatch_norm_0/batchnorm/mul/ReadVariableOpReadVariableOp;model_1_ebatch_norm_0_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
#model_1/ebatch_norm_0/batchnorm/mulMul)model_1/ebatch_norm_0/batchnorm/Rsqrt:y:0:model_1/ebatch_norm_0/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
%model_1/ebatch_norm_0/batchnorm/mul_1Mul"model_1/encoder_0/BiasAdd:output:0'model_1/ebatch_norm_0/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
0model_1/ebatch_norm_0/batchnorm/ReadVariableOp_1ReadVariableOp9model_1_ebatch_norm_0_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0�
%model_1/ebatch_norm_0/batchnorm/mul_2Mul8model_1/ebatch_norm_0/batchnorm/ReadVariableOp_1:value:0'model_1/ebatch_norm_0/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
0model_1/ebatch_norm_0/batchnorm/ReadVariableOp_2ReadVariableOp9model_1_ebatch_norm_0_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0�
#model_1/ebatch_norm_0/batchnorm/subSub8model_1/ebatch_norm_0/batchnorm/ReadVariableOp_2:value:0)model_1/ebatch_norm_0/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
%model_1/ebatch_norm_0/batchnorm/add_1AddV2)model_1/ebatch_norm_0/batchnorm/mul_1:z:0'model_1/ebatch_norm_0/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Po
model_1/elnorm_0/ShapeShape)model_1/ebatch_norm_0/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
$model_1/elnorm_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model_1/elnorm_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model_1/elnorm_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model_1/elnorm_0/strided_sliceStridedSlicemodel_1/elnorm_0/Shape:output:0-model_1/elnorm_0/strided_slice/stack:output:0/model_1/elnorm_0/strided_slice/stack_1:output:0/model_1/elnorm_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
model_1/elnorm_0/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/elnorm_0/mulMulmodel_1/elnorm_0/mul/x:output:0'model_1/elnorm_0/strided_slice:output:0*
T0*
_output_shapes
: p
&model_1/elnorm_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(model_1/elnorm_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(model_1/elnorm_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 model_1/elnorm_0/strided_slice_1StridedSlicemodel_1/elnorm_0/Shape:output:0/model_1/elnorm_0/strided_slice_1/stack:output:01model_1/elnorm_0/strided_slice_1/stack_1:output:01model_1/elnorm_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
model_1/elnorm_0/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/elnorm_0/mul_1Mul!model_1/elnorm_0/mul_1/x:output:0)model_1/elnorm_0/strided_slice_1:output:0*
T0*
_output_shapes
: b
 model_1/elnorm_0/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :b
 model_1/elnorm_0/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
model_1/elnorm_0/Reshape/shapePack)model_1/elnorm_0/Reshape/shape/0:output:0model_1/elnorm_0/mul:z:0model_1/elnorm_0/mul_1:z:0)model_1/elnorm_0/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
model_1/elnorm_0/ReshapeReshape)model_1/ebatch_norm_0/batchnorm/add_1:z:0'model_1/elnorm_0/Reshape/shape:output:0*
T0*/
_output_shapes
:���������Pl
model_1/elnorm_0/ones/packedPackmodel_1/elnorm_0/mul:z:0*
N*
T0*
_output_shapes
:`
model_1/elnorm_0/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_1/elnorm_0/onesFill%model_1/elnorm_0/ones/packed:output:0$model_1/elnorm_0/ones/Const:output:0*
T0*#
_output_shapes
:���������m
model_1/elnorm_0/zeros/packedPackmodel_1/elnorm_0/mul:z:0*
N*
T0*
_output_shapes
:a
model_1/elnorm_0/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_1/elnorm_0/zerosFill&model_1/elnorm_0/zeros/packed:output:0%model_1/elnorm_0/zeros/Const:output:0*
T0*#
_output_shapes
:���������Y
model_1/elnorm_0/ConstConst*
_output_shapes
: *
dtype0*
valueB [
model_1/elnorm_0/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
!model_1/elnorm_0/FusedBatchNormV3FusedBatchNormV3!model_1/elnorm_0/Reshape:output:0model_1/elnorm_0/ones:output:0model_1/elnorm_0/zeros:output:0model_1/elnorm_0/Const:output:0!model_1/elnorm_0/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
model_1/elnorm_0/Reshape_1Reshape%model_1/elnorm_0/FusedBatchNormV3:y:0model_1/elnorm_0/Shape:output:0*
T0*'
_output_shapes
:���������P�
%model_1/elnorm_0/mul_2/ReadVariableOpReadVariableOp.model_1_elnorm_0_mul_2_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_1/elnorm_0/mul_2Mul#model_1/elnorm_0/Reshape_1:output:0-model_1/elnorm_0/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
#model_1/elnorm_0/add/ReadVariableOpReadVariableOp,model_1_elnorm_0_add_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_1/elnorm_0/addAddV2model_1/elnorm_0/mul_2:z:0+model_1/elnorm_0/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
'model_1/encoder_1/MatMul/ReadVariableOpReadVariableOp0model_1_encoder_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
model_1/encoder_1/MatMulMatMulmodel_1/elnorm_0/add:z:0/model_1/encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
(model_1/encoder_1/BiasAdd/ReadVariableOpReadVariableOp1model_1_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_1/encoder_1/BiasAddBiasAdd"model_1/encoder_1/MatMul:product:00model_1/encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
.model_1/ebatch_norm_1/batchnorm/ReadVariableOpReadVariableOp7model_1_ebatch_norm_1_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0j
%model_1/ebatch_norm_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#model_1/ebatch_norm_1/batchnorm/addAddV26model_1/ebatch_norm_1/batchnorm/ReadVariableOp:value:0.model_1/ebatch_norm_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:P|
%model_1/ebatch_norm_1/batchnorm/RsqrtRsqrt'model_1/ebatch_norm_1/batchnorm/add:z:0*
T0*
_output_shapes
:P�
2model_1/ebatch_norm_1/batchnorm/mul/ReadVariableOpReadVariableOp;model_1_ebatch_norm_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
#model_1/ebatch_norm_1/batchnorm/mulMul)model_1/ebatch_norm_1/batchnorm/Rsqrt:y:0:model_1/ebatch_norm_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
%model_1/ebatch_norm_1/batchnorm/mul_1Mul"model_1/encoder_1/BiasAdd:output:0'model_1/ebatch_norm_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
0model_1/ebatch_norm_1/batchnorm/ReadVariableOp_1ReadVariableOp9model_1_ebatch_norm_1_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0�
%model_1/ebatch_norm_1/batchnorm/mul_2Mul8model_1/ebatch_norm_1/batchnorm/ReadVariableOp_1:value:0'model_1/ebatch_norm_1/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
0model_1/ebatch_norm_1/batchnorm/ReadVariableOp_2ReadVariableOp9model_1_ebatch_norm_1_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0�
#model_1/ebatch_norm_1/batchnorm/subSub8model_1/ebatch_norm_1/batchnorm/ReadVariableOp_2:value:0)model_1/ebatch_norm_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
%model_1/ebatch_norm_1/batchnorm/add_1AddV2)model_1/ebatch_norm_1/batchnorm/mul_1:z:0'model_1/ebatch_norm_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Po
model_1/elnorm_1/ShapeShape)model_1/ebatch_norm_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
$model_1/elnorm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model_1/elnorm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model_1/elnorm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model_1/elnorm_1/strided_sliceStridedSlicemodel_1/elnorm_1/Shape:output:0-model_1/elnorm_1/strided_slice/stack:output:0/model_1/elnorm_1/strided_slice/stack_1:output:0/model_1/elnorm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
model_1/elnorm_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/elnorm_1/mulMulmodel_1/elnorm_1/mul/x:output:0'model_1/elnorm_1/strided_slice:output:0*
T0*
_output_shapes
: p
&model_1/elnorm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(model_1/elnorm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(model_1/elnorm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 model_1/elnorm_1/strided_slice_1StridedSlicemodel_1/elnorm_1/Shape:output:0/model_1/elnorm_1/strided_slice_1/stack:output:01model_1/elnorm_1/strided_slice_1/stack_1:output:01model_1/elnorm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
model_1/elnorm_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/elnorm_1/mul_1Mul!model_1/elnorm_1/mul_1/x:output:0)model_1/elnorm_1/strided_slice_1:output:0*
T0*
_output_shapes
: b
 model_1/elnorm_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :b
 model_1/elnorm_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
model_1/elnorm_1/Reshape/shapePack)model_1/elnorm_1/Reshape/shape/0:output:0model_1/elnorm_1/mul:z:0model_1/elnorm_1/mul_1:z:0)model_1/elnorm_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
model_1/elnorm_1/ReshapeReshape)model_1/ebatch_norm_1/batchnorm/add_1:z:0'model_1/elnorm_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������Pl
model_1/elnorm_1/ones/packedPackmodel_1/elnorm_1/mul:z:0*
N*
T0*
_output_shapes
:`
model_1/elnorm_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_1/elnorm_1/onesFill%model_1/elnorm_1/ones/packed:output:0$model_1/elnorm_1/ones/Const:output:0*
T0*#
_output_shapes
:���������m
model_1/elnorm_1/zeros/packedPackmodel_1/elnorm_1/mul:z:0*
N*
T0*
_output_shapes
:a
model_1/elnorm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_1/elnorm_1/zerosFill&model_1/elnorm_1/zeros/packed:output:0%model_1/elnorm_1/zeros/Const:output:0*
T0*#
_output_shapes
:���������Y
model_1/elnorm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB [
model_1/elnorm_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
!model_1/elnorm_1/FusedBatchNormV3FusedBatchNormV3!model_1/elnorm_1/Reshape:output:0model_1/elnorm_1/ones:output:0model_1/elnorm_1/zeros:output:0model_1/elnorm_1/Const:output:0!model_1/elnorm_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
model_1/elnorm_1/Reshape_1Reshape%model_1/elnorm_1/FusedBatchNormV3:y:0model_1/elnorm_1/Shape:output:0*
T0*'
_output_shapes
:���������P�
%model_1/elnorm_1/mul_2/ReadVariableOpReadVariableOp.model_1_elnorm_1_mul_2_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_1/elnorm_1/mul_2Mul#model_1/elnorm_1/Reshape_1:output:0-model_1/elnorm_1/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
#model_1/elnorm_1/add/ReadVariableOpReadVariableOp,model_1_elnorm_1_add_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_1/elnorm_1/addAddV2model_1/elnorm_1/mul_2:z:0+model_1/elnorm_1/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
'model_1/encoder_2/MatMul/ReadVariableOpReadVariableOp0model_1_encoder_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
model_1/encoder_2/MatMulMatMulmodel_1/elnorm_1/add:z:0/model_1/encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
(model_1/encoder_2/BiasAdd/ReadVariableOpReadVariableOp1model_1_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_1/encoder_2/BiasAddBiasAdd"model_1/encoder_2/MatMul:product:00model_1/encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
.model_1/ebatch_norm_2/batchnorm/ReadVariableOpReadVariableOp7model_1_ebatch_norm_2_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0j
%model_1/ebatch_norm_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#model_1/ebatch_norm_2/batchnorm/addAddV26model_1/ebatch_norm_2/batchnorm/ReadVariableOp:value:0.model_1/ebatch_norm_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:P|
%model_1/ebatch_norm_2/batchnorm/RsqrtRsqrt'model_1/ebatch_norm_2/batchnorm/add:z:0*
T0*
_output_shapes
:P�
2model_1/ebatch_norm_2/batchnorm/mul/ReadVariableOpReadVariableOp;model_1_ebatch_norm_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
#model_1/ebatch_norm_2/batchnorm/mulMul)model_1/ebatch_norm_2/batchnorm/Rsqrt:y:0:model_1/ebatch_norm_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
%model_1/ebatch_norm_2/batchnorm/mul_1Mul"model_1/encoder_2/BiasAdd:output:0'model_1/ebatch_norm_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
0model_1/ebatch_norm_2/batchnorm/ReadVariableOp_1ReadVariableOp9model_1_ebatch_norm_2_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0�
%model_1/ebatch_norm_2/batchnorm/mul_2Mul8model_1/ebatch_norm_2/batchnorm/ReadVariableOp_1:value:0'model_1/ebatch_norm_2/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
0model_1/ebatch_norm_2/batchnorm/ReadVariableOp_2ReadVariableOp9model_1_ebatch_norm_2_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0�
#model_1/ebatch_norm_2/batchnorm/subSub8model_1/ebatch_norm_2/batchnorm/ReadVariableOp_2:value:0)model_1/ebatch_norm_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
%model_1/ebatch_norm_2/batchnorm/add_1AddV2)model_1/ebatch_norm_2/batchnorm/mul_1:z:0'model_1/ebatch_norm_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Po
model_1/elnorm_2/ShapeShape)model_1/ebatch_norm_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
$model_1/elnorm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model_1/elnorm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model_1/elnorm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model_1/elnorm_2/strided_sliceStridedSlicemodel_1/elnorm_2/Shape:output:0-model_1/elnorm_2/strided_slice/stack:output:0/model_1/elnorm_2/strided_slice/stack_1:output:0/model_1/elnorm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
model_1/elnorm_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/elnorm_2/mulMulmodel_1/elnorm_2/mul/x:output:0'model_1/elnorm_2/strided_slice:output:0*
T0*
_output_shapes
: p
&model_1/elnorm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(model_1/elnorm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(model_1/elnorm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 model_1/elnorm_2/strided_slice_1StridedSlicemodel_1/elnorm_2/Shape:output:0/model_1/elnorm_2/strided_slice_1/stack:output:01model_1/elnorm_2/strided_slice_1/stack_1:output:01model_1/elnorm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
model_1/elnorm_2/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/elnorm_2/mul_1Mul!model_1/elnorm_2/mul_1/x:output:0)model_1/elnorm_2/strided_slice_1:output:0*
T0*
_output_shapes
: b
 model_1/elnorm_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :b
 model_1/elnorm_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
model_1/elnorm_2/Reshape/shapePack)model_1/elnorm_2/Reshape/shape/0:output:0model_1/elnorm_2/mul:z:0model_1/elnorm_2/mul_1:z:0)model_1/elnorm_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
model_1/elnorm_2/ReshapeReshape)model_1/ebatch_norm_2/batchnorm/add_1:z:0'model_1/elnorm_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������Pl
model_1/elnorm_2/ones/packedPackmodel_1/elnorm_2/mul:z:0*
N*
T0*
_output_shapes
:`
model_1/elnorm_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_1/elnorm_2/onesFill%model_1/elnorm_2/ones/packed:output:0$model_1/elnorm_2/ones/Const:output:0*
T0*#
_output_shapes
:���������m
model_1/elnorm_2/zeros/packedPackmodel_1/elnorm_2/mul:z:0*
N*
T0*
_output_shapes
:a
model_1/elnorm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_1/elnorm_2/zerosFill&model_1/elnorm_2/zeros/packed:output:0%model_1/elnorm_2/zeros/Const:output:0*
T0*#
_output_shapes
:���������Y
model_1/elnorm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB [
model_1/elnorm_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
!model_1/elnorm_2/FusedBatchNormV3FusedBatchNormV3!model_1/elnorm_2/Reshape:output:0model_1/elnorm_2/ones:output:0model_1/elnorm_2/zeros:output:0model_1/elnorm_2/Const:output:0!model_1/elnorm_2/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
model_1/elnorm_2/Reshape_1Reshape%model_1/elnorm_2/FusedBatchNormV3:y:0model_1/elnorm_2/Shape:output:0*
T0*'
_output_shapes
:���������P�
%model_1/elnorm_2/mul_2/ReadVariableOpReadVariableOp.model_1_elnorm_2_mul_2_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_1/elnorm_2/mul_2Mul#model_1/elnorm_2/Reshape_1:output:0-model_1/elnorm_2/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
#model_1/elnorm_2/add/ReadVariableOpReadVariableOp,model_1_elnorm_2_add_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_1/elnorm_2/addAddV2model_1/elnorm_2/mul_2:z:0+model_1/elnorm_2/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
'model_1/encoder_3/MatMul/ReadVariableOpReadVariableOp0model_1_encoder_3_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
model_1/encoder_3/MatMulMatMulmodel_1/elnorm_2/add:z:0/model_1/encoder_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
(model_1/encoder_3/BiasAdd/ReadVariableOpReadVariableOp1model_1_encoder_3_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_1/encoder_3/BiasAddBiasAdd"model_1/encoder_3/MatMul:product:00model_1/encoder_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
.model_1/ebatch_norm_3/batchnorm/ReadVariableOpReadVariableOp7model_1_ebatch_norm_3_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0j
%model_1/ebatch_norm_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#model_1/ebatch_norm_3/batchnorm/addAddV26model_1/ebatch_norm_3/batchnorm/ReadVariableOp:value:0.model_1/ebatch_norm_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:P|
%model_1/ebatch_norm_3/batchnorm/RsqrtRsqrt'model_1/ebatch_norm_3/batchnorm/add:z:0*
T0*
_output_shapes
:P�
2model_1/ebatch_norm_3/batchnorm/mul/ReadVariableOpReadVariableOp;model_1_ebatch_norm_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
#model_1/ebatch_norm_3/batchnorm/mulMul)model_1/ebatch_norm_3/batchnorm/Rsqrt:y:0:model_1/ebatch_norm_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
%model_1/ebatch_norm_3/batchnorm/mul_1Mul"model_1/encoder_3/BiasAdd:output:0'model_1/ebatch_norm_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
0model_1/ebatch_norm_3/batchnorm/ReadVariableOp_1ReadVariableOp9model_1_ebatch_norm_3_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0�
%model_1/ebatch_norm_3/batchnorm/mul_2Mul8model_1/ebatch_norm_3/batchnorm/ReadVariableOp_1:value:0'model_1/ebatch_norm_3/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
0model_1/ebatch_norm_3/batchnorm/ReadVariableOp_2ReadVariableOp9model_1_ebatch_norm_3_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0�
#model_1/ebatch_norm_3/batchnorm/subSub8model_1/ebatch_norm_3/batchnorm/ReadVariableOp_2:value:0)model_1/ebatch_norm_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
%model_1/ebatch_norm_3/batchnorm/add_1AddV2)model_1/ebatch_norm_3/batchnorm/mul_1:z:0'model_1/ebatch_norm_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Po
model_1/elnorm_3/ShapeShape)model_1/ebatch_norm_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
$model_1/elnorm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model_1/elnorm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model_1/elnorm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model_1/elnorm_3/strided_sliceStridedSlicemodel_1/elnorm_3/Shape:output:0-model_1/elnorm_3/strided_slice/stack:output:0/model_1/elnorm_3/strided_slice/stack_1:output:0/model_1/elnorm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
model_1/elnorm_3/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/elnorm_3/mulMulmodel_1/elnorm_3/mul/x:output:0'model_1/elnorm_3/strided_slice:output:0*
T0*
_output_shapes
: p
&model_1/elnorm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(model_1/elnorm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(model_1/elnorm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 model_1/elnorm_3/strided_slice_1StridedSlicemodel_1/elnorm_3/Shape:output:0/model_1/elnorm_3/strided_slice_1/stack:output:01model_1/elnorm_3/strided_slice_1/stack_1:output:01model_1/elnorm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
model_1/elnorm_3/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/elnorm_3/mul_1Mul!model_1/elnorm_3/mul_1/x:output:0)model_1/elnorm_3/strided_slice_1:output:0*
T0*
_output_shapes
: b
 model_1/elnorm_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :b
 model_1/elnorm_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
model_1/elnorm_3/Reshape/shapePack)model_1/elnorm_3/Reshape/shape/0:output:0model_1/elnorm_3/mul:z:0model_1/elnorm_3/mul_1:z:0)model_1/elnorm_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
model_1/elnorm_3/ReshapeReshape)model_1/ebatch_norm_3/batchnorm/add_1:z:0'model_1/elnorm_3/Reshape/shape:output:0*
T0*/
_output_shapes
:���������Pl
model_1/elnorm_3/ones/packedPackmodel_1/elnorm_3/mul:z:0*
N*
T0*
_output_shapes
:`
model_1/elnorm_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_1/elnorm_3/onesFill%model_1/elnorm_3/ones/packed:output:0$model_1/elnorm_3/ones/Const:output:0*
T0*#
_output_shapes
:���������m
model_1/elnorm_3/zeros/packedPackmodel_1/elnorm_3/mul:z:0*
N*
T0*
_output_shapes
:a
model_1/elnorm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_1/elnorm_3/zerosFill&model_1/elnorm_3/zeros/packed:output:0%model_1/elnorm_3/zeros/Const:output:0*
T0*#
_output_shapes
:���������Y
model_1/elnorm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB [
model_1/elnorm_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
!model_1/elnorm_3/FusedBatchNormV3FusedBatchNormV3!model_1/elnorm_3/Reshape:output:0model_1/elnorm_3/ones:output:0model_1/elnorm_3/zeros:output:0model_1/elnorm_3/Const:output:0!model_1/elnorm_3/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
model_1/elnorm_3/Reshape_1Reshape%model_1/elnorm_3/FusedBatchNormV3:y:0model_1/elnorm_3/Shape:output:0*
T0*'
_output_shapes
:���������P�
%model_1/elnorm_3/mul_2/ReadVariableOpReadVariableOp.model_1_elnorm_3_mul_2_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_1/elnorm_3/mul_2Mul#model_1/elnorm_3/Reshape_1:output:0-model_1/elnorm_3/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
#model_1/elnorm_3/add/ReadVariableOpReadVariableOp,model_1_elnorm_3_add_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_1/elnorm_3/addAddV2model_1/elnorm_3/mul_2:z:0+model_1/elnorm_3/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
)model_1/ebottleneck/MatMul/ReadVariableOpReadVariableOp2model_1_ebottleneck_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0�
model_1/ebottleneck/MatMulMatMulmodel_1/elnorm_3/add:z:01model_1/ebottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*model_1/ebottleneck/BiasAdd/ReadVariableOpReadVariableOp3model_1_ebottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1/ebottleneck/BiasAddBiasAdd$model_1/ebottleneck/MatMul:product:02model_1/ebottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
model_1/ebottleneck/ReluRelu$model_1/ebottleneck/BiasAdd:output:0*
T0*'
_output_shapes
:���������f
model_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
model_1/flatten/ReshapeReshape&model_1/ebottleneck/Relu:activations:0model_1/flatten/Const:output:0*
T0*'
_output_shapes
:����������
'model_1/regress_1/MatMul/ReadVariableOpReadVariableOp0model_1_regress_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
model_1/regress_1/MatMulMatMul model_1/flatten/Reshape:output:0/model_1/regress_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
(model_1/regress_1/BiasAdd/ReadVariableOpReadVariableOp1model_1_regress_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
model_1/regress_1/BiasAddBiasAdd"model_1/regress_1/MatMul:product:00model_1/regress_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2t
model_1/regress_1/ReluRelu"model_1/regress_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
+model_1/reg_norm_1/batchnorm/ReadVariableOpReadVariableOp4model_1_reg_norm_1_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0g
"model_1/reg_norm_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 model_1/reg_norm_1/batchnorm/addAddV23model_1/reg_norm_1/batchnorm/ReadVariableOp:value:0+model_1/reg_norm_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2v
"model_1/reg_norm_1/batchnorm/RsqrtRsqrt$model_1/reg_norm_1/batchnorm/add:z:0*
T0*
_output_shapes
:2�
/model_1/reg_norm_1/batchnorm/mul/ReadVariableOpReadVariableOp8model_1_reg_norm_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0�
 model_1/reg_norm_1/batchnorm/mulMul&model_1/reg_norm_1/batchnorm/Rsqrt:y:07model_1/reg_norm_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2�
"model_1/reg_norm_1/batchnorm/mul_1Mul$model_1/regress_1/Relu:activations:0$model_1/reg_norm_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2�
-model_1/reg_norm_1/batchnorm/ReadVariableOp_1ReadVariableOp6model_1_reg_norm_1_batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0�
"model_1/reg_norm_1/batchnorm/mul_2Mul5model_1/reg_norm_1/batchnorm/ReadVariableOp_1:value:0$model_1/reg_norm_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2�
-model_1/reg_norm_1/batchnorm/ReadVariableOp_2ReadVariableOp6model_1_reg_norm_1_batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0�
 model_1/reg_norm_1/batchnorm/subSub5model_1/reg_norm_1/batchnorm/ReadVariableOp_2:value:0&model_1/reg_norm_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2�
"model_1/reg_norm_1/batchnorm/add_1AddV2&model_1/reg_norm_1/batchnorm/mul_1:z:0$model_1/reg_norm_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2~
model_1/dropout/IdentityIdentity&model_1/reg_norm_1/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2�
#model_1/dense/MatMul/ReadVariableOpReadVariableOp,model_1_dense_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
model_1/dense/MatMulMatMul!model_1/dropout/Identity:output:0+model_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model_1/dense/BiasAdd/ReadVariableOpReadVariableOp-model_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1/dense/BiasAddBiasAddmodel_1/dense/MatMul:product:0,model_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������m
IdentityIdentitymodel_1/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^model_1/dense/BiasAdd/ReadVariableOp$^model_1/dense/MatMul/ReadVariableOp/^model_1/ebatch_norm_0/batchnorm/ReadVariableOp1^model_1/ebatch_norm_0/batchnorm/ReadVariableOp_11^model_1/ebatch_norm_0/batchnorm/ReadVariableOp_23^model_1/ebatch_norm_0/batchnorm/mul/ReadVariableOp/^model_1/ebatch_norm_1/batchnorm/ReadVariableOp1^model_1/ebatch_norm_1/batchnorm/ReadVariableOp_11^model_1/ebatch_norm_1/batchnorm/ReadVariableOp_23^model_1/ebatch_norm_1/batchnorm/mul/ReadVariableOp/^model_1/ebatch_norm_2/batchnorm/ReadVariableOp1^model_1/ebatch_norm_2/batchnorm/ReadVariableOp_11^model_1/ebatch_norm_2/batchnorm/ReadVariableOp_23^model_1/ebatch_norm_2/batchnorm/mul/ReadVariableOp/^model_1/ebatch_norm_3/batchnorm/ReadVariableOp1^model_1/ebatch_norm_3/batchnorm/ReadVariableOp_11^model_1/ebatch_norm_3/batchnorm/ReadVariableOp_23^model_1/ebatch_norm_3/batchnorm/mul/ReadVariableOp+^model_1/ebottleneck/BiasAdd/ReadVariableOp*^model_1/ebottleneck/MatMul/ReadVariableOp$^model_1/elnorm_0/add/ReadVariableOp&^model_1/elnorm_0/mul_2/ReadVariableOp$^model_1/elnorm_1/add/ReadVariableOp&^model_1/elnorm_1/mul_2/ReadVariableOp$^model_1/elnorm_2/add/ReadVariableOp&^model_1/elnorm_2/mul_2/ReadVariableOp$^model_1/elnorm_3/add/ReadVariableOp&^model_1/elnorm_3/mul_2/ReadVariableOp)^model_1/encoder_0/BiasAdd/ReadVariableOp(^model_1/encoder_0/MatMul/ReadVariableOp)^model_1/encoder_1/BiasAdd/ReadVariableOp(^model_1/encoder_1/MatMul/ReadVariableOp)^model_1/encoder_2/BiasAdd/ReadVariableOp(^model_1/encoder_2/MatMul/ReadVariableOp)^model_1/encoder_3/BiasAdd/ReadVariableOp(^model_1/encoder_3/MatMul/ReadVariableOp,^model_1/reg_norm_1/batchnorm/ReadVariableOp.^model_1/reg_norm_1/batchnorm/ReadVariableOp_1.^model_1/reg_norm_1/batchnorm/ReadVariableOp_20^model_1/reg_norm_1/batchnorm/mul/ReadVariableOp)^model_1/regress_1/BiasAdd/ReadVariableOp(^model_1/regress_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$model_1/dense/BiasAdd/ReadVariableOp$model_1/dense/BiasAdd/ReadVariableOp2J
#model_1/dense/MatMul/ReadVariableOp#model_1/dense/MatMul/ReadVariableOp2`
.model_1/ebatch_norm_0/batchnorm/ReadVariableOp.model_1/ebatch_norm_0/batchnorm/ReadVariableOp2d
0model_1/ebatch_norm_0/batchnorm/ReadVariableOp_10model_1/ebatch_norm_0/batchnorm/ReadVariableOp_12d
0model_1/ebatch_norm_0/batchnorm/ReadVariableOp_20model_1/ebatch_norm_0/batchnorm/ReadVariableOp_22h
2model_1/ebatch_norm_0/batchnorm/mul/ReadVariableOp2model_1/ebatch_norm_0/batchnorm/mul/ReadVariableOp2`
.model_1/ebatch_norm_1/batchnorm/ReadVariableOp.model_1/ebatch_norm_1/batchnorm/ReadVariableOp2d
0model_1/ebatch_norm_1/batchnorm/ReadVariableOp_10model_1/ebatch_norm_1/batchnorm/ReadVariableOp_12d
0model_1/ebatch_norm_1/batchnorm/ReadVariableOp_20model_1/ebatch_norm_1/batchnorm/ReadVariableOp_22h
2model_1/ebatch_norm_1/batchnorm/mul/ReadVariableOp2model_1/ebatch_norm_1/batchnorm/mul/ReadVariableOp2`
.model_1/ebatch_norm_2/batchnorm/ReadVariableOp.model_1/ebatch_norm_2/batchnorm/ReadVariableOp2d
0model_1/ebatch_norm_2/batchnorm/ReadVariableOp_10model_1/ebatch_norm_2/batchnorm/ReadVariableOp_12d
0model_1/ebatch_norm_2/batchnorm/ReadVariableOp_20model_1/ebatch_norm_2/batchnorm/ReadVariableOp_22h
2model_1/ebatch_norm_2/batchnorm/mul/ReadVariableOp2model_1/ebatch_norm_2/batchnorm/mul/ReadVariableOp2`
.model_1/ebatch_norm_3/batchnorm/ReadVariableOp.model_1/ebatch_norm_3/batchnorm/ReadVariableOp2d
0model_1/ebatch_norm_3/batchnorm/ReadVariableOp_10model_1/ebatch_norm_3/batchnorm/ReadVariableOp_12d
0model_1/ebatch_norm_3/batchnorm/ReadVariableOp_20model_1/ebatch_norm_3/batchnorm/ReadVariableOp_22h
2model_1/ebatch_norm_3/batchnorm/mul/ReadVariableOp2model_1/ebatch_norm_3/batchnorm/mul/ReadVariableOp2X
*model_1/ebottleneck/BiasAdd/ReadVariableOp*model_1/ebottleneck/BiasAdd/ReadVariableOp2V
)model_1/ebottleneck/MatMul/ReadVariableOp)model_1/ebottleneck/MatMul/ReadVariableOp2J
#model_1/elnorm_0/add/ReadVariableOp#model_1/elnorm_0/add/ReadVariableOp2N
%model_1/elnorm_0/mul_2/ReadVariableOp%model_1/elnorm_0/mul_2/ReadVariableOp2J
#model_1/elnorm_1/add/ReadVariableOp#model_1/elnorm_1/add/ReadVariableOp2N
%model_1/elnorm_1/mul_2/ReadVariableOp%model_1/elnorm_1/mul_2/ReadVariableOp2J
#model_1/elnorm_2/add/ReadVariableOp#model_1/elnorm_2/add/ReadVariableOp2N
%model_1/elnorm_2/mul_2/ReadVariableOp%model_1/elnorm_2/mul_2/ReadVariableOp2J
#model_1/elnorm_3/add/ReadVariableOp#model_1/elnorm_3/add/ReadVariableOp2N
%model_1/elnorm_3/mul_2/ReadVariableOp%model_1/elnorm_3/mul_2/ReadVariableOp2T
(model_1/encoder_0/BiasAdd/ReadVariableOp(model_1/encoder_0/BiasAdd/ReadVariableOp2R
'model_1/encoder_0/MatMul/ReadVariableOp'model_1/encoder_0/MatMul/ReadVariableOp2T
(model_1/encoder_1/BiasAdd/ReadVariableOp(model_1/encoder_1/BiasAdd/ReadVariableOp2R
'model_1/encoder_1/MatMul/ReadVariableOp'model_1/encoder_1/MatMul/ReadVariableOp2T
(model_1/encoder_2/BiasAdd/ReadVariableOp(model_1/encoder_2/BiasAdd/ReadVariableOp2R
'model_1/encoder_2/MatMul/ReadVariableOp'model_1/encoder_2/MatMul/ReadVariableOp2T
(model_1/encoder_3/BiasAdd/ReadVariableOp(model_1/encoder_3/BiasAdd/ReadVariableOp2R
'model_1/encoder_3/MatMul/ReadVariableOp'model_1/encoder_3/MatMul/ReadVariableOp2Z
+model_1/reg_norm_1/batchnorm/ReadVariableOp+model_1/reg_norm_1/batchnorm/ReadVariableOp2^
-model_1/reg_norm_1/batchnorm/ReadVariableOp_1-model_1/reg_norm_1/batchnorm/ReadVariableOp_12^
-model_1/reg_norm_1/batchnorm/ReadVariableOp_2-model_1/reg_norm_1/batchnorm/ReadVariableOp_22b
/model_1/reg_norm_1/batchnorm/mul/ReadVariableOp/model_1/reg_norm_1/batchnorm/mul/ReadVariableOp2T
(model_1/regress_1/BiasAdd/ReadVariableOp(model_1/regress_1/BiasAdd/ReadVariableOp2R
'model_1/regress_1/MatMul/ReadVariableOp'model_1/regress_1/MatMul/ReadVariableOp:R N
'
_output_shapes
:���������
#
_user_specified_name	intensity
�
�
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1357957

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Pz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�%
�
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1357840

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Ph
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�
D__inference_model_1_layer_call_and_return_conditional_losses_1359370
	intensity#
encoder_0_1359237:P
encoder_0_1359239:P#
ebatch_norm_0_1359242:P#
ebatch_norm_0_1359244:P#
ebatch_norm_0_1359246:P#
ebatch_norm_0_1359248:P
elnorm_0_1359251:P
elnorm_0_1359253:P#
encoder_1_1359256:PP
encoder_1_1359258:P#
ebatch_norm_1_1359261:P#
ebatch_norm_1_1359263:P#
ebatch_norm_1_1359265:P#
ebatch_norm_1_1359267:P
elnorm_1_1359270:P
elnorm_1_1359272:P#
encoder_2_1359275:PP
encoder_2_1359277:P#
ebatch_norm_2_1359280:P#
ebatch_norm_2_1359282:P#
ebatch_norm_2_1359284:P#
ebatch_norm_2_1359286:P
elnorm_2_1359289:P
elnorm_2_1359291:P#
encoder_3_1359294:PP
encoder_3_1359296:P#
ebatch_norm_3_1359299:P#
ebatch_norm_3_1359301:P#
ebatch_norm_3_1359303:P#
ebatch_norm_3_1359305:P
elnorm_3_1359308:P
elnorm_3_1359310:P%
ebottleneck_1359313:P!
ebottleneck_1359315:#
regress_1_1359319:2
regress_1_1359321:2 
reg_norm_1_1359324:2 
reg_norm_1_1359326:2 
reg_norm_1_1359328:2 
reg_norm_1_1359330:2
dense_1359334:2
dense_1359336:
identity��dense/StatefulPartitionedCall�%ebatch_norm_0/StatefulPartitionedCall�%ebatch_norm_1/StatefulPartitionedCall�%ebatch_norm_2/StatefulPartitionedCall�%ebatch_norm_3/StatefulPartitionedCall�#ebottleneck/StatefulPartitionedCall� elnorm_0/StatefulPartitionedCall� elnorm_1/StatefulPartitionedCall� elnorm_2/StatefulPartitionedCall� elnorm_3/StatefulPartitionedCall�!encoder_0/StatefulPartitionedCall�2encoder_0/kernel/Regularizer/Square/ReadVariableOp�!encoder_1/StatefulPartitionedCall�2encoder_1/kernel/Regularizer/Square/ReadVariableOp�!encoder_2/StatefulPartitionedCall�2encoder_2/kernel/Regularizer/Square/ReadVariableOp�!encoder_3/StatefulPartitionedCall�2encoder_3/kernel/Regularizer/Square/ReadVariableOp�"reg_norm_1/StatefulPartitionedCall�!regress_1/StatefulPartitionedCall�2regress_1/kernel/Regularizer/Square/ReadVariableOp�
!encoder_0/StatefulPartitionedCallStatefulPartitionedCall	intensityencoder_0_1359237encoder_0_1359239*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_0_layer_call_and_return_conditional_losses_1358202�
%ebatch_norm_0/StatefulPartitionedCallStatefulPartitionedCall*encoder_0/StatefulPartitionedCall:output:0ebatch_norm_0_1359242ebatch_norm_0_1359244ebatch_norm_0_1359246ebatch_norm_0_1359248*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1357793�
 elnorm_0/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_0/StatefulPartitionedCall:output:0elnorm_0_1359251elnorm_0_1359253*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_0_layer_call_and_return_conditional_losses_1358259�
!encoder_1/StatefulPartitionedCallStatefulPartitionedCall)elnorm_0/StatefulPartitionedCall:output:0encoder_1_1359256encoder_1_1359258*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_1_layer_call_and_return_conditional_losses_1358281�
%ebatch_norm_1/StatefulPartitionedCallStatefulPartitionedCall*encoder_1/StatefulPartitionedCall:output:0ebatch_norm_1_1359261ebatch_norm_1_1359263ebatch_norm_1_1359265ebatch_norm_1_1359267*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1357875�
 elnorm_1/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_1/StatefulPartitionedCall:output:0elnorm_1_1359270elnorm_1_1359272*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_1_layer_call_and_return_conditional_losses_1358338�
!encoder_2/StatefulPartitionedCallStatefulPartitionedCall)elnorm_1/StatefulPartitionedCall:output:0encoder_2_1359275encoder_2_1359277*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_2_layer_call_and_return_conditional_losses_1358360�
%ebatch_norm_2/StatefulPartitionedCallStatefulPartitionedCall*encoder_2/StatefulPartitionedCall:output:0ebatch_norm_2_1359280ebatch_norm_2_1359282ebatch_norm_2_1359284ebatch_norm_2_1359286*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1357957�
 elnorm_2/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_2/StatefulPartitionedCall:output:0elnorm_2_1359289elnorm_2_1359291*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_2_layer_call_and_return_conditional_losses_1358417�
!encoder_3/StatefulPartitionedCallStatefulPartitionedCall)elnorm_2/StatefulPartitionedCall:output:0encoder_3_1359294encoder_3_1359296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_3_layer_call_and_return_conditional_losses_1358439�
%ebatch_norm_3/StatefulPartitionedCallStatefulPartitionedCall*encoder_3/StatefulPartitionedCall:output:0ebatch_norm_3_1359299ebatch_norm_3_1359301ebatch_norm_3_1359303ebatch_norm_3_1359305*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1358039�
 elnorm_3/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_3/StatefulPartitionedCall:output:0elnorm_3_1359308elnorm_3_1359310*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_3_layer_call_and_return_conditional_losses_1358496�
#ebottleneck/StatefulPartitionedCallStatefulPartitionedCall)elnorm_3/StatefulPartitionedCall:output:0ebottleneck_1359313ebottleneck_1359315*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_ebottleneck_layer_call_and_return_conditional_losses_1358513�
flatten/PartitionedCallPartitionedCall,ebottleneck/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1358525�
!regress_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0regress_1_1359319regress_1_1359321*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_regress_1_layer_call_and_return_conditional_losses_1358544�
"reg_norm_1/StatefulPartitionedCallStatefulPartitionedCall*regress_1/StatefulPartitionedCall:output:0reg_norm_1_1359324reg_norm_1_1359326reg_norm_1_1359328reg_norm_1_1359330*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1358121�
dropout/PartitionedCallPartitionedCall+reg_norm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1358564�
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1359334dense_1359336*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1358576�
2encoder_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_0_1359237*
_output_shapes

:P*
dtype0�
#encoder_0/kernel/Regularizer/SquareSquare:encoder_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Ps
"encoder_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_0/kernel/Regularizer/SumSum'encoder_0/kernel/Regularizer/Square:y:0+encoder_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_0/kernel/Regularizer/mulMul+encoder_0/kernel/Regularizer/mul/x:output:0)encoder_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_1_1359256*
_output_shapes

:PP*
dtype0�
#encoder_1/kernel/Regularizer/SquareSquare:encoder_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_1/kernel/Regularizer/SumSum'encoder_1/kernel/Regularizer/Square:y:0+encoder_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_1/kernel/Regularizer/mulMul+encoder_1/kernel/Regularizer/mul/x:output:0)encoder_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_2_1359275*
_output_shapes

:PP*
dtype0�
#encoder_2/kernel/Regularizer/SquareSquare:encoder_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_2/kernel/Regularizer/SumSum'encoder_2/kernel/Regularizer/Square:y:0+encoder_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_2/kernel/Regularizer/mulMul+encoder_2/kernel/Regularizer/mul/x:output:0)encoder_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_3_1359294*
_output_shapes

:PP*
dtype0�
#encoder_3/kernel/Regularizer/SquareSquare:encoder_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_3/kernel/Regularizer/SumSum'encoder_3/kernel/Regularizer/Square:y:0+encoder_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_3/kernel/Regularizer/mulMul+encoder_3/kernel/Regularizer/mul/x:output:0)encoder_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2regress_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpregress_1_1359319*
_output_shapes

:2*
dtype0�
#regress_1/kernel/Regularizer/SquareSquare:regress_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2s
"regress_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 regress_1/kernel/Regularizer/SumSum'regress_1/kernel/Regularizer/Square:y:0+regress_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"regress_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 regress_1/kernel/Regularizer/mulMul+regress_1/kernel/Regularizer/mul/x:output:0)regress_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall&^ebatch_norm_0/StatefulPartitionedCall&^ebatch_norm_1/StatefulPartitionedCall&^ebatch_norm_2/StatefulPartitionedCall&^ebatch_norm_3/StatefulPartitionedCall$^ebottleneck/StatefulPartitionedCall!^elnorm_0/StatefulPartitionedCall!^elnorm_1/StatefulPartitionedCall!^elnorm_2/StatefulPartitionedCall!^elnorm_3/StatefulPartitionedCall"^encoder_0/StatefulPartitionedCall3^encoder_0/kernel/Regularizer/Square/ReadVariableOp"^encoder_1/StatefulPartitionedCall3^encoder_1/kernel/Regularizer/Square/ReadVariableOp"^encoder_2/StatefulPartitionedCall3^encoder_2/kernel/Regularizer/Square/ReadVariableOp"^encoder_3/StatefulPartitionedCall3^encoder_3/kernel/Regularizer/Square/ReadVariableOp#^reg_norm_1/StatefulPartitionedCall"^regress_1/StatefulPartitionedCall3^regress_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%ebatch_norm_0/StatefulPartitionedCall%ebatch_norm_0/StatefulPartitionedCall2N
%ebatch_norm_1/StatefulPartitionedCall%ebatch_norm_1/StatefulPartitionedCall2N
%ebatch_norm_2/StatefulPartitionedCall%ebatch_norm_2/StatefulPartitionedCall2N
%ebatch_norm_3/StatefulPartitionedCall%ebatch_norm_3/StatefulPartitionedCall2J
#ebottleneck/StatefulPartitionedCall#ebottleneck/StatefulPartitionedCall2D
 elnorm_0/StatefulPartitionedCall elnorm_0/StatefulPartitionedCall2D
 elnorm_1/StatefulPartitionedCall elnorm_1/StatefulPartitionedCall2D
 elnorm_2/StatefulPartitionedCall elnorm_2/StatefulPartitionedCall2D
 elnorm_3/StatefulPartitionedCall elnorm_3/StatefulPartitionedCall2F
!encoder_0/StatefulPartitionedCall!encoder_0/StatefulPartitionedCall2h
2encoder_0/kernel/Regularizer/Square/ReadVariableOp2encoder_0/kernel/Regularizer/Square/ReadVariableOp2F
!encoder_1/StatefulPartitionedCall!encoder_1/StatefulPartitionedCall2h
2encoder_1/kernel/Regularizer/Square/ReadVariableOp2encoder_1/kernel/Regularizer/Square/ReadVariableOp2F
!encoder_2/StatefulPartitionedCall!encoder_2/StatefulPartitionedCall2h
2encoder_2/kernel/Regularizer/Square/ReadVariableOp2encoder_2/kernel/Regularizer/Square/ReadVariableOp2F
!encoder_3/StatefulPartitionedCall!encoder_3/StatefulPartitionedCall2h
2encoder_3/kernel/Regularizer/Square/ReadVariableOp2encoder_3/kernel/Regularizer/Square/ReadVariableOp2H
"reg_norm_1/StatefulPartitionedCall"reg_norm_1/StatefulPartitionedCall2F
!regress_1/StatefulPartitionedCall!regress_1/StatefulPartitionedCall2h
2regress_1/kernel/Regularizer/Square/ReadVariableOp2regress_1/kernel/Regularizer/Square/ReadVariableOp:R N
'
_output_shapes
:���������
#
_user_specified_name	intensity
�
�
+__inference_encoder_0_layer_call_fn_1360545

inputs
unknown:P
	unknown_0:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_0_layer_call_and_return_conditional_losses_1358202o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_encoder_1_layer_call_and_return_conditional_losses_1358281

inputs0
matmul_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2encoder_1/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
2encoder_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_1/kernel/Regularizer/SquareSquare:encoder_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_1/kernel/Regularizer/SumSum'encoder_1/kernel/Regularizer/Square:y:0+encoder_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_1/kernel/Regularizer/mulMul+encoder_1/kernel/Regularizer/mul/x:output:0)encoder_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^encoder_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2encoder_1/kernel/Regularizer/Square/ReadVariableOp2encoder_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_1361379M
;encoder_3_kernel_regularizer_square_readvariableop_resource:PP
identity��2encoder_3/kernel/Regularizer/Square/ReadVariableOp�
2encoder_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;encoder_3_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_3/kernel/Regularizer/SquareSquare:encoder_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_3/kernel/Regularizer/SumSum'encoder_3/kernel/Regularizer/Square:y:0+encoder_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_3/kernel/Regularizer/mulMul+encoder_3/kernel/Regularizer/mul/x:output:0)encoder_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$encoder_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^encoder_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2encoder_3/kernel/Regularizer/Square/ReadVariableOp2encoder_3/kernel/Regularizer/Square/ReadVariableOp
�
�
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1360919

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Pz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
F__inference_regress_1_layer_call_and_return_conditional_losses_1361217

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2regress_1/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2�
2regress_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
#regress_1/kernel/Regularizer/SquareSquare:regress_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2s
"regress_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 regress_1/kernel/Regularizer/SumSum'regress_1/kernel/Regularizer/Square:y:0+regress_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"regress_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 regress_1/kernel/Regularizer/mulMul+regress_1/kernel/Regularizer/mul/x:output:0)regress_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^regress_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2regress_1/kernel/Regularizer/Square/ReadVariableOp2regress_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_encoder_3_layer_call_fn_1361013

inputs
unknown:PP
	unknown_0:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_3_layer_call_and_return_conditional_losses_1358439o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_1361368M
;encoder_2_kernel_regularizer_square_readvariableop_resource:PP
identity��2encoder_2/kernel/Regularizer/Square/ReadVariableOp�
2encoder_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;encoder_2_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_2/kernel/Regularizer/SquareSquare:encoder_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_2/kernel/Regularizer/SumSum'encoder_2/kernel/Regularizer/Square:y:0+encoder_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_2/kernel/Regularizer/mulMul+encoder_2/kernel/Regularizer/mul/x:output:0)encoder_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$encoder_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^encoder_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2encoder_2/kernel/Regularizer/Square/ReadVariableOp2encoder_2/kernel/Regularizer/Square/ReadVariableOp
�
E
)__inference_flatten_layer_call_fn_1361185

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1358525`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1358086

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Ph
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�%
�
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1357922

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Ph
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_1361346M
;encoder_0_kernel_regularizer_square_readvariableop_resource:P
identity��2encoder_0/kernel/Regularizer/Square/ReadVariableOp�
2encoder_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;encoder_0_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:P*
dtype0�
#encoder_0/kernel/Regularizer/SquareSquare:encoder_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Ps
"encoder_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_0/kernel/Regularizer/SumSum'encoder_0/kernel/Regularizer/Square:y:0+encoder_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_0/kernel/Regularizer/mulMul+encoder_0/kernel/Regularizer/mul/x:output:0)encoder_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$encoder_0/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^encoder_0/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2encoder_0/kernel/Regularizer/Square/ReadVariableOp2encoder_0/kernel/Regularizer/Square/ReadVariableOp
�
�	
)__inference_model_1_layer_call_fn_1359234
	intensity
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P

unknown_10:P

unknown_11:P

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:PP

unknown_16:P

unknown_17:P

unknown_18:P

unknown_19:P

unknown_20:P

unknown_21:P

unknown_22:P

unknown_23:PP

unknown_24:P

unknown_25:P

unknown_26:P

unknown_27:P

unknown_28:P

unknown_29:P

unknown_30:P

unknown_31:P

unknown_32:

unknown_33:2

unknown_34:2

unknown_35:2

unknown_36:2

unknown_37:2

unknown_38:2

unknown_39:2

unknown_40:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	intensityunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 	
 !"#$'()**6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1359058o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:���������
#
_user_specified_name	intensity
�	
�
B__inference_dense_layer_call_and_return_conditional_losses_1361335

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
/__inference_ebatch_norm_0_layer_call_fn_1360587

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1357840o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
*__inference_elnorm_1_layer_call_fn_1360806

inputs
unknown:P
	unknown_0:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_1_layer_call_and_return_conditional_losses_1358338o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
F__inference_encoder_3_layer_call_and_return_conditional_losses_1361029

inputs0
matmul_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2encoder_3/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
2encoder_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_3/kernel/Regularizer/SquareSquare:encoder_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_3/kernel/Regularizer/SumSum'encoder_3/kernel/Regularizer/Square:y:0+encoder_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_3/kernel/Regularizer/mulMul+encoder_3/kernel/Regularizer/mul/x:output:0)encoder_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^encoder_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2encoder_3/kernel/Regularizer/Square/ReadVariableOp2encoder_3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
-__inference_ebottleneck_layer_call_fn_1361169

inputs
unknown:P
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_ebottleneck_layer_call_and_return_conditional_losses_1358513o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�"
D__inference_model_1_layer_call_and_return_conditional_losses_1360154

inputs:
(encoder_0_matmul_readvariableop_resource:P7
)encoder_0_biasadd_readvariableop_resource:P=
/ebatch_norm_0_batchnorm_readvariableop_resource:PA
3ebatch_norm_0_batchnorm_mul_readvariableop_resource:P?
1ebatch_norm_0_batchnorm_readvariableop_1_resource:P?
1ebatch_norm_0_batchnorm_readvariableop_2_resource:P4
&elnorm_0_mul_2_readvariableop_resource:P2
$elnorm_0_add_readvariableop_resource:P:
(encoder_1_matmul_readvariableop_resource:PP7
)encoder_1_biasadd_readvariableop_resource:P=
/ebatch_norm_1_batchnorm_readvariableop_resource:PA
3ebatch_norm_1_batchnorm_mul_readvariableop_resource:P?
1ebatch_norm_1_batchnorm_readvariableop_1_resource:P?
1ebatch_norm_1_batchnorm_readvariableop_2_resource:P4
&elnorm_1_mul_2_readvariableop_resource:P2
$elnorm_1_add_readvariableop_resource:P:
(encoder_2_matmul_readvariableop_resource:PP7
)encoder_2_biasadd_readvariableop_resource:P=
/ebatch_norm_2_batchnorm_readvariableop_resource:PA
3ebatch_norm_2_batchnorm_mul_readvariableop_resource:P?
1ebatch_norm_2_batchnorm_readvariableop_1_resource:P?
1ebatch_norm_2_batchnorm_readvariableop_2_resource:P4
&elnorm_2_mul_2_readvariableop_resource:P2
$elnorm_2_add_readvariableop_resource:P:
(encoder_3_matmul_readvariableop_resource:PP7
)encoder_3_biasadd_readvariableop_resource:P=
/ebatch_norm_3_batchnorm_readvariableop_resource:PA
3ebatch_norm_3_batchnorm_mul_readvariableop_resource:P?
1ebatch_norm_3_batchnorm_readvariableop_1_resource:P?
1ebatch_norm_3_batchnorm_readvariableop_2_resource:P4
&elnorm_3_mul_2_readvariableop_resource:P2
$elnorm_3_add_readvariableop_resource:P<
*ebottleneck_matmul_readvariableop_resource:P9
+ebottleneck_biasadd_readvariableop_resource::
(regress_1_matmul_readvariableop_resource:27
)regress_1_biasadd_readvariableop_resource:2:
,reg_norm_1_batchnorm_readvariableop_resource:2>
0reg_norm_1_batchnorm_mul_readvariableop_resource:2<
.reg_norm_1_batchnorm_readvariableop_1_resource:2<
.reg_norm_1_batchnorm_readvariableop_2_resource:26
$dense_matmul_readvariableop_resource:23
%dense_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�&ebatch_norm_0/batchnorm/ReadVariableOp�(ebatch_norm_0/batchnorm/ReadVariableOp_1�(ebatch_norm_0/batchnorm/ReadVariableOp_2�*ebatch_norm_0/batchnorm/mul/ReadVariableOp�&ebatch_norm_1/batchnorm/ReadVariableOp�(ebatch_norm_1/batchnorm/ReadVariableOp_1�(ebatch_norm_1/batchnorm/ReadVariableOp_2�*ebatch_norm_1/batchnorm/mul/ReadVariableOp�&ebatch_norm_2/batchnorm/ReadVariableOp�(ebatch_norm_2/batchnorm/ReadVariableOp_1�(ebatch_norm_2/batchnorm/ReadVariableOp_2�*ebatch_norm_2/batchnorm/mul/ReadVariableOp�&ebatch_norm_3/batchnorm/ReadVariableOp�(ebatch_norm_3/batchnorm/ReadVariableOp_1�(ebatch_norm_3/batchnorm/ReadVariableOp_2�*ebatch_norm_3/batchnorm/mul/ReadVariableOp�"ebottleneck/BiasAdd/ReadVariableOp�!ebottleneck/MatMul/ReadVariableOp�elnorm_0/add/ReadVariableOp�elnorm_0/mul_2/ReadVariableOp�elnorm_1/add/ReadVariableOp�elnorm_1/mul_2/ReadVariableOp�elnorm_2/add/ReadVariableOp�elnorm_2/mul_2/ReadVariableOp�elnorm_3/add/ReadVariableOp�elnorm_3/mul_2/ReadVariableOp� encoder_0/BiasAdd/ReadVariableOp�encoder_0/MatMul/ReadVariableOp�2encoder_0/kernel/Regularizer/Square/ReadVariableOp� encoder_1/BiasAdd/ReadVariableOp�encoder_1/MatMul/ReadVariableOp�2encoder_1/kernel/Regularizer/Square/ReadVariableOp� encoder_2/BiasAdd/ReadVariableOp�encoder_2/MatMul/ReadVariableOp�2encoder_2/kernel/Regularizer/Square/ReadVariableOp� encoder_3/BiasAdd/ReadVariableOp�encoder_3/MatMul/ReadVariableOp�2encoder_3/kernel/Regularizer/Square/ReadVariableOp�#reg_norm_1/batchnorm/ReadVariableOp�%reg_norm_1/batchnorm/ReadVariableOp_1�%reg_norm_1/batchnorm/ReadVariableOp_2�'reg_norm_1/batchnorm/mul/ReadVariableOp� regress_1/BiasAdd/ReadVariableOp�regress_1/MatMul/ReadVariableOp�2regress_1/kernel/Regularizer/Square/ReadVariableOp�
encoder_0/MatMul/ReadVariableOpReadVariableOp(encoder_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0}
encoder_0/MatMulMatMulinputs'encoder_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
 encoder_0/BiasAdd/ReadVariableOpReadVariableOp)encoder_0_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
encoder_0/BiasAddBiasAddencoder_0/MatMul:product:0(encoder_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
&ebatch_norm_0/batchnorm/ReadVariableOpReadVariableOp/ebatch_norm_0_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0b
ebatch_norm_0/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
ebatch_norm_0/batchnorm/addAddV2.ebatch_norm_0/batchnorm/ReadVariableOp:value:0&ebatch_norm_0/batchnorm/add/y:output:0*
T0*
_output_shapes
:Pl
ebatch_norm_0/batchnorm/RsqrtRsqrtebatch_norm_0/batchnorm/add:z:0*
T0*
_output_shapes
:P�
*ebatch_norm_0/batchnorm/mul/ReadVariableOpReadVariableOp3ebatch_norm_0_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_0/batchnorm/mulMul!ebatch_norm_0/batchnorm/Rsqrt:y:02ebatch_norm_0/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
ebatch_norm_0/batchnorm/mul_1Mulencoder_0/BiasAdd:output:0ebatch_norm_0/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
(ebatch_norm_0/batchnorm/ReadVariableOp_1ReadVariableOp1ebatch_norm_0_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_0/batchnorm/mul_2Mul0ebatch_norm_0/batchnorm/ReadVariableOp_1:value:0ebatch_norm_0/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
(ebatch_norm_0/batchnorm/ReadVariableOp_2ReadVariableOp1ebatch_norm_0_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_0/batchnorm/subSub0ebatch_norm_0/batchnorm/ReadVariableOp_2:value:0!ebatch_norm_0/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
ebatch_norm_0/batchnorm/add_1AddV2!ebatch_norm_0/batchnorm/mul_1:z:0ebatch_norm_0/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P_
elnorm_0/ShapeShape!ebatch_norm_0/batchnorm/add_1:z:0*
T0*
_output_shapes
:f
elnorm_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
elnorm_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
elnorm_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_0/strided_sliceStridedSliceelnorm_0/Shape:output:0%elnorm_0/strided_slice/stack:output:0'elnorm_0/strided_slice/stack_1:output:0'elnorm_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
elnorm_0/mul/xConst*
_output_shapes
: *
dtype0*
value	B :n
elnorm_0/mulMulelnorm_0/mul/x:output:0elnorm_0/strided_slice:output:0*
T0*
_output_shapes
: h
elnorm_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_0/strided_slice_1StridedSliceelnorm_0/Shape:output:0'elnorm_0/strided_slice_1/stack:output:0)elnorm_0/strided_slice_1/stack_1:output:0)elnorm_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
elnorm_0/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :t
elnorm_0/mul_1Mulelnorm_0/mul_1/x:output:0!elnorm_0/strided_slice_1:output:0*
T0*
_output_shapes
: Z
elnorm_0/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Z
elnorm_0/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
elnorm_0/Reshape/shapePack!elnorm_0/Reshape/shape/0:output:0elnorm_0/mul:z:0elnorm_0/mul_1:z:0!elnorm_0/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
elnorm_0/ReshapeReshape!ebatch_norm_0/batchnorm/add_1:z:0elnorm_0/Reshape/shape:output:0*
T0*/
_output_shapes
:���������P\
elnorm_0/ones/packedPackelnorm_0/mul:z:0*
N*
T0*
_output_shapes
:X
elnorm_0/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
elnorm_0/onesFillelnorm_0/ones/packed:output:0elnorm_0/ones/Const:output:0*
T0*#
_output_shapes
:���������]
elnorm_0/zeros/packedPackelnorm_0/mul:z:0*
N*
T0*
_output_shapes
:Y
elnorm_0/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
elnorm_0/zerosFillelnorm_0/zeros/packed:output:0elnorm_0/zeros/Const:output:0*
T0*#
_output_shapes
:���������Q
elnorm_0/ConstConst*
_output_shapes
: *
dtype0*
valueB S
elnorm_0/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
elnorm_0/FusedBatchNormV3FusedBatchNormV3elnorm_0/Reshape:output:0elnorm_0/ones:output:0elnorm_0/zeros:output:0elnorm_0/Const:output:0elnorm_0/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
elnorm_0/Reshape_1Reshapeelnorm_0/FusedBatchNormV3:y:0elnorm_0/Shape:output:0*
T0*'
_output_shapes
:���������P�
elnorm_0/mul_2/ReadVariableOpReadVariableOp&elnorm_0_mul_2_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_0/mul_2Mulelnorm_0/Reshape_1:output:0%elnorm_0/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P|
elnorm_0/add/ReadVariableOpReadVariableOp$elnorm_0_add_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_0/addAddV2elnorm_0/mul_2:z:0#elnorm_0/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
encoder_1/MatMul/ReadVariableOpReadVariableOp(encoder_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
encoder_1/MatMulMatMulelnorm_0/add:z:0'encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
 encoder_1/BiasAdd/ReadVariableOpReadVariableOp)encoder_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
encoder_1/BiasAddBiasAddencoder_1/MatMul:product:0(encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
&ebatch_norm_1/batchnorm/ReadVariableOpReadVariableOp/ebatch_norm_1_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0b
ebatch_norm_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
ebatch_norm_1/batchnorm/addAddV2.ebatch_norm_1/batchnorm/ReadVariableOp:value:0&ebatch_norm_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:Pl
ebatch_norm_1/batchnorm/RsqrtRsqrtebatch_norm_1/batchnorm/add:z:0*
T0*
_output_shapes
:P�
*ebatch_norm_1/batchnorm/mul/ReadVariableOpReadVariableOp3ebatch_norm_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_1/batchnorm/mulMul!ebatch_norm_1/batchnorm/Rsqrt:y:02ebatch_norm_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
ebatch_norm_1/batchnorm/mul_1Mulencoder_1/BiasAdd:output:0ebatch_norm_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
(ebatch_norm_1/batchnorm/ReadVariableOp_1ReadVariableOp1ebatch_norm_1_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_1/batchnorm/mul_2Mul0ebatch_norm_1/batchnorm/ReadVariableOp_1:value:0ebatch_norm_1/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
(ebatch_norm_1/batchnorm/ReadVariableOp_2ReadVariableOp1ebatch_norm_1_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_1/batchnorm/subSub0ebatch_norm_1/batchnorm/ReadVariableOp_2:value:0!ebatch_norm_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
ebatch_norm_1/batchnorm/add_1AddV2!ebatch_norm_1/batchnorm/mul_1:z:0ebatch_norm_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P_
elnorm_1/ShapeShape!ebatch_norm_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:f
elnorm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
elnorm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
elnorm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_1/strided_sliceStridedSliceelnorm_1/Shape:output:0%elnorm_1/strided_slice/stack:output:0'elnorm_1/strided_slice/stack_1:output:0'elnorm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
elnorm_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :n
elnorm_1/mulMulelnorm_1/mul/x:output:0elnorm_1/strided_slice:output:0*
T0*
_output_shapes
: h
elnorm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_1/strided_slice_1StridedSliceelnorm_1/Shape:output:0'elnorm_1/strided_slice_1/stack:output:0)elnorm_1/strided_slice_1/stack_1:output:0)elnorm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
elnorm_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :t
elnorm_1/mul_1Mulelnorm_1/mul_1/x:output:0!elnorm_1/strided_slice_1:output:0*
T0*
_output_shapes
: Z
elnorm_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Z
elnorm_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
elnorm_1/Reshape/shapePack!elnorm_1/Reshape/shape/0:output:0elnorm_1/mul:z:0elnorm_1/mul_1:z:0!elnorm_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
elnorm_1/ReshapeReshape!ebatch_norm_1/batchnorm/add_1:z:0elnorm_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������P\
elnorm_1/ones/packedPackelnorm_1/mul:z:0*
N*
T0*
_output_shapes
:X
elnorm_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
elnorm_1/onesFillelnorm_1/ones/packed:output:0elnorm_1/ones/Const:output:0*
T0*#
_output_shapes
:���������]
elnorm_1/zeros/packedPackelnorm_1/mul:z:0*
N*
T0*
_output_shapes
:Y
elnorm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
elnorm_1/zerosFillelnorm_1/zeros/packed:output:0elnorm_1/zeros/Const:output:0*
T0*#
_output_shapes
:���������Q
elnorm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB S
elnorm_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
elnorm_1/FusedBatchNormV3FusedBatchNormV3elnorm_1/Reshape:output:0elnorm_1/ones:output:0elnorm_1/zeros:output:0elnorm_1/Const:output:0elnorm_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
elnorm_1/Reshape_1Reshapeelnorm_1/FusedBatchNormV3:y:0elnorm_1/Shape:output:0*
T0*'
_output_shapes
:���������P�
elnorm_1/mul_2/ReadVariableOpReadVariableOp&elnorm_1_mul_2_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_1/mul_2Mulelnorm_1/Reshape_1:output:0%elnorm_1/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P|
elnorm_1/add/ReadVariableOpReadVariableOp$elnorm_1_add_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_1/addAddV2elnorm_1/mul_2:z:0#elnorm_1/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
encoder_2/MatMul/ReadVariableOpReadVariableOp(encoder_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
encoder_2/MatMulMatMulelnorm_1/add:z:0'encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
 encoder_2/BiasAdd/ReadVariableOpReadVariableOp)encoder_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
encoder_2/BiasAddBiasAddencoder_2/MatMul:product:0(encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
&ebatch_norm_2/batchnorm/ReadVariableOpReadVariableOp/ebatch_norm_2_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0b
ebatch_norm_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
ebatch_norm_2/batchnorm/addAddV2.ebatch_norm_2/batchnorm/ReadVariableOp:value:0&ebatch_norm_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:Pl
ebatch_norm_2/batchnorm/RsqrtRsqrtebatch_norm_2/batchnorm/add:z:0*
T0*
_output_shapes
:P�
*ebatch_norm_2/batchnorm/mul/ReadVariableOpReadVariableOp3ebatch_norm_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_2/batchnorm/mulMul!ebatch_norm_2/batchnorm/Rsqrt:y:02ebatch_norm_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
ebatch_norm_2/batchnorm/mul_1Mulencoder_2/BiasAdd:output:0ebatch_norm_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
(ebatch_norm_2/batchnorm/ReadVariableOp_1ReadVariableOp1ebatch_norm_2_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_2/batchnorm/mul_2Mul0ebatch_norm_2/batchnorm/ReadVariableOp_1:value:0ebatch_norm_2/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
(ebatch_norm_2/batchnorm/ReadVariableOp_2ReadVariableOp1ebatch_norm_2_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_2/batchnorm/subSub0ebatch_norm_2/batchnorm/ReadVariableOp_2:value:0!ebatch_norm_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
ebatch_norm_2/batchnorm/add_1AddV2!ebatch_norm_2/batchnorm/mul_1:z:0ebatch_norm_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P_
elnorm_2/ShapeShape!ebatch_norm_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:f
elnorm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
elnorm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
elnorm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_2/strided_sliceStridedSliceelnorm_2/Shape:output:0%elnorm_2/strided_slice/stack:output:0'elnorm_2/strided_slice/stack_1:output:0'elnorm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
elnorm_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :n
elnorm_2/mulMulelnorm_2/mul/x:output:0elnorm_2/strided_slice:output:0*
T0*
_output_shapes
: h
elnorm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_2/strided_slice_1StridedSliceelnorm_2/Shape:output:0'elnorm_2/strided_slice_1/stack:output:0)elnorm_2/strided_slice_1/stack_1:output:0)elnorm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
elnorm_2/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :t
elnorm_2/mul_1Mulelnorm_2/mul_1/x:output:0!elnorm_2/strided_slice_1:output:0*
T0*
_output_shapes
: Z
elnorm_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Z
elnorm_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
elnorm_2/Reshape/shapePack!elnorm_2/Reshape/shape/0:output:0elnorm_2/mul:z:0elnorm_2/mul_1:z:0!elnorm_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
elnorm_2/ReshapeReshape!ebatch_norm_2/batchnorm/add_1:z:0elnorm_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������P\
elnorm_2/ones/packedPackelnorm_2/mul:z:0*
N*
T0*
_output_shapes
:X
elnorm_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
elnorm_2/onesFillelnorm_2/ones/packed:output:0elnorm_2/ones/Const:output:0*
T0*#
_output_shapes
:���������]
elnorm_2/zeros/packedPackelnorm_2/mul:z:0*
N*
T0*
_output_shapes
:Y
elnorm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
elnorm_2/zerosFillelnorm_2/zeros/packed:output:0elnorm_2/zeros/Const:output:0*
T0*#
_output_shapes
:���������Q
elnorm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB S
elnorm_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
elnorm_2/FusedBatchNormV3FusedBatchNormV3elnorm_2/Reshape:output:0elnorm_2/ones:output:0elnorm_2/zeros:output:0elnorm_2/Const:output:0elnorm_2/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
elnorm_2/Reshape_1Reshapeelnorm_2/FusedBatchNormV3:y:0elnorm_2/Shape:output:0*
T0*'
_output_shapes
:���������P�
elnorm_2/mul_2/ReadVariableOpReadVariableOp&elnorm_2_mul_2_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_2/mul_2Mulelnorm_2/Reshape_1:output:0%elnorm_2/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P|
elnorm_2/add/ReadVariableOpReadVariableOp$elnorm_2_add_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_2/addAddV2elnorm_2/mul_2:z:0#elnorm_2/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
encoder_3/MatMul/ReadVariableOpReadVariableOp(encoder_3_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
encoder_3/MatMulMatMulelnorm_2/add:z:0'encoder_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
 encoder_3/BiasAdd/ReadVariableOpReadVariableOp)encoder_3_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
encoder_3/BiasAddBiasAddencoder_3/MatMul:product:0(encoder_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
&ebatch_norm_3/batchnorm/ReadVariableOpReadVariableOp/ebatch_norm_3_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0b
ebatch_norm_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
ebatch_norm_3/batchnorm/addAddV2.ebatch_norm_3/batchnorm/ReadVariableOp:value:0&ebatch_norm_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:Pl
ebatch_norm_3/batchnorm/RsqrtRsqrtebatch_norm_3/batchnorm/add:z:0*
T0*
_output_shapes
:P�
*ebatch_norm_3/batchnorm/mul/ReadVariableOpReadVariableOp3ebatch_norm_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_3/batchnorm/mulMul!ebatch_norm_3/batchnorm/Rsqrt:y:02ebatch_norm_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
ebatch_norm_3/batchnorm/mul_1Mulencoder_3/BiasAdd:output:0ebatch_norm_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
(ebatch_norm_3/batchnorm/ReadVariableOp_1ReadVariableOp1ebatch_norm_3_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_3/batchnorm/mul_2Mul0ebatch_norm_3/batchnorm/ReadVariableOp_1:value:0ebatch_norm_3/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
(ebatch_norm_3/batchnorm/ReadVariableOp_2ReadVariableOp1ebatch_norm_3_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_3/batchnorm/subSub0ebatch_norm_3/batchnorm/ReadVariableOp_2:value:0!ebatch_norm_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
ebatch_norm_3/batchnorm/add_1AddV2!ebatch_norm_3/batchnorm/mul_1:z:0ebatch_norm_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P_
elnorm_3/ShapeShape!ebatch_norm_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:f
elnorm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
elnorm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
elnorm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_3/strided_sliceStridedSliceelnorm_3/Shape:output:0%elnorm_3/strided_slice/stack:output:0'elnorm_3/strided_slice/stack_1:output:0'elnorm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
elnorm_3/mul/xConst*
_output_shapes
: *
dtype0*
value	B :n
elnorm_3/mulMulelnorm_3/mul/x:output:0elnorm_3/strided_slice:output:0*
T0*
_output_shapes
: h
elnorm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_3/strided_slice_1StridedSliceelnorm_3/Shape:output:0'elnorm_3/strided_slice_1/stack:output:0)elnorm_3/strided_slice_1/stack_1:output:0)elnorm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
elnorm_3/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :t
elnorm_3/mul_1Mulelnorm_3/mul_1/x:output:0!elnorm_3/strided_slice_1:output:0*
T0*
_output_shapes
: Z
elnorm_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Z
elnorm_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
elnorm_3/Reshape/shapePack!elnorm_3/Reshape/shape/0:output:0elnorm_3/mul:z:0elnorm_3/mul_1:z:0!elnorm_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
elnorm_3/ReshapeReshape!ebatch_norm_3/batchnorm/add_1:z:0elnorm_3/Reshape/shape:output:0*
T0*/
_output_shapes
:���������P\
elnorm_3/ones/packedPackelnorm_3/mul:z:0*
N*
T0*
_output_shapes
:X
elnorm_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
elnorm_3/onesFillelnorm_3/ones/packed:output:0elnorm_3/ones/Const:output:0*
T0*#
_output_shapes
:���������]
elnorm_3/zeros/packedPackelnorm_3/mul:z:0*
N*
T0*
_output_shapes
:Y
elnorm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
elnorm_3/zerosFillelnorm_3/zeros/packed:output:0elnorm_3/zeros/Const:output:0*
T0*#
_output_shapes
:���������Q
elnorm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB S
elnorm_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
elnorm_3/FusedBatchNormV3FusedBatchNormV3elnorm_3/Reshape:output:0elnorm_3/ones:output:0elnorm_3/zeros:output:0elnorm_3/Const:output:0elnorm_3/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
elnorm_3/Reshape_1Reshapeelnorm_3/FusedBatchNormV3:y:0elnorm_3/Shape:output:0*
T0*'
_output_shapes
:���������P�
elnorm_3/mul_2/ReadVariableOpReadVariableOp&elnorm_3_mul_2_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_3/mul_2Mulelnorm_3/Reshape_1:output:0%elnorm_3/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P|
elnorm_3/add/ReadVariableOpReadVariableOp$elnorm_3_add_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_3/addAddV2elnorm_3/mul_2:z:0#elnorm_3/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
!ebottleneck/MatMul/ReadVariableOpReadVariableOp*ebottleneck_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0�
ebottleneck/MatMulMatMulelnorm_3/add:z:0)ebottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"ebottleneck/BiasAdd/ReadVariableOpReadVariableOp+ebottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
ebottleneck/BiasAddBiasAddebottleneck/MatMul:product:0*ebottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
ebottleneck/ReluReluebottleneck/BiasAdd:output:0*
T0*'
_output_shapes
:���������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapeebottleneck/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:����������
regress_1/MatMul/ReadVariableOpReadVariableOp(regress_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
regress_1/MatMulMatMulflatten/Reshape:output:0'regress_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
 regress_1/BiasAdd/ReadVariableOpReadVariableOp)regress_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
regress_1/BiasAddBiasAddregress_1/MatMul:product:0(regress_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2d
regress_1/ReluReluregress_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
#reg_norm_1/batchnorm/ReadVariableOpReadVariableOp,reg_norm_1_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0_
reg_norm_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
reg_norm_1/batchnorm/addAddV2+reg_norm_1/batchnorm/ReadVariableOp:value:0#reg_norm_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2f
reg_norm_1/batchnorm/RsqrtRsqrtreg_norm_1/batchnorm/add:z:0*
T0*
_output_shapes
:2�
'reg_norm_1/batchnorm/mul/ReadVariableOpReadVariableOp0reg_norm_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0�
reg_norm_1/batchnorm/mulMulreg_norm_1/batchnorm/Rsqrt:y:0/reg_norm_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2�
reg_norm_1/batchnorm/mul_1Mulregress_1/Relu:activations:0reg_norm_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2�
%reg_norm_1/batchnorm/ReadVariableOp_1ReadVariableOp.reg_norm_1_batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0�
reg_norm_1/batchnorm/mul_2Mul-reg_norm_1/batchnorm/ReadVariableOp_1:value:0reg_norm_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2�
%reg_norm_1/batchnorm/ReadVariableOp_2ReadVariableOp.reg_norm_1_batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0�
reg_norm_1/batchnorm/subSub-reg_norm_1/batchnorm/ReadVariableOp_2:value:0reg_norm_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2�
reg_norm_1/batchnorm/add_1AddV2reg_norm_1/batchnorm/mul_1:z:0reg_norm_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2n
dropout/IdentityIdentityreg_norm_1/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2encoder_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(encoder_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0�
#encoder_0/kernel/Regularizer/SquareSquare:encoder_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Ps
"encoder_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_0/kernel/Regularizer/SumSum'encoder_0/kernel/Regularizer/Square:y:0+encoder_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_0/kernel/Regularizer/mulMul+encoder_0/kernel/Regularizer/mul/x:output:0)encoder_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(encoder_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_1/kernel/Regularizer/SquareSquare:encoder_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_1/kernel/Regularizer/SumSum'encoder_1/kernel/Regularizer/Square:y:0+encoder_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_1/kernel/Regularizer/mulMul+encoder_1/kernel/Regularizer/mul/x:output:0)encoder_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(encoder_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_2/kernel/Regularizer/SquareSquare:encoder_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_2/kernel/Regularizer/SumSum'encoder_2/kernel/Regularizer/Square:y:0+encoder_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_2/kernel/Regularizer/mulMul+encoder_2/kernel/Regularizer/mul/x:output:0)encoder_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(encoder_3_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_3/kernel/Regularizer/SquareSquare:encoder_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_3/kernel/Regularizer/SumSum'encoder_3/kernel/Regularizer/Square:y:0+encoder_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_3/kernel/Regularizer/mulMul+encoder_3/kernel/Regularizer/mul/x:output:0)encoder_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2regress_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(regress_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
#regress_1/kernel/Regularizer/SquareSquare:regress_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2s
"regress_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 regress_1/kernel/Regularizer/SumSum'regress_1/kernel/Regularizer/Square:y:0+regress_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"regress_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 regress_1/kernel/Regularizer/mulMul+regress_1/kernel/Regularizer/mul/x:output:0)regress_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp'^ebatch_norm_0/batchnorm/ReadVariableOp)^ebatch_norm_0/batchnorm/ReadVariableOp_1)^ebatch_norm_0/batchnorm/ReadVariableOp_2+^ebatch_norm_0/batchnorm/mul/ReadVariableOp'^ebatch_norm_1/batchnorm/ReadVariableOp)^ebatch_norm_1/batchnorm/ReadVariableOp_1)^ebatch_norm_1/batchnorm/ReadVariableOp_2+^ebatch_norm_1/batchnorm/mul/ReadVariableOp'^ebatch_norm_2/batchnorm/ReadVariableOp)^ebatch_norm_2/batchnorm/ReadVariableOp_1)^ebatch_norm_2/batchnorm/ReadVariableOp_2+^ebatch_norm_2/batchnorm/mul/ReadVariableOp'^ebatch_norm_3/batchnorm/ReadVariableOp)^ebatch_norm_3/batchnorm/ReadVariableOp_1)^ebatch_norm_3/batchnorm/ReadVariableOp_2+^ebatch_norm_3/batchnorm/mul/ReadVariableOp#^ebottleneck/BiasAdd/ReadVariableOp"^ebottleneck/MatMul/ReadVariableOp^elnorm_0/add/ReadVariableOp^elnorm_0/mul_2/ReadVariableOp^elnorm_1/add/ReadVariableOp^elnorm_1/mul_2/ReadVariableOp^elnorm_2/add/ReadVariableOp^elnorm_2/mul_2/ReadVariableOp^elnorm_3/add/ReadVariableOp^elnorm_3/mul_2/ReadVariableOp!^encoder_0/BiasAdd/ReadVariableOp ^encoder_0/MatMul/ReadVariableOp3^encoder_0/kernel/Regularizer/Square/ReadVariableOp!^encoder_1/BiasAdd/ReadVariableOp ^encoder_1/MatMul/ReadVariableOp3^encoder_1/kernel/Regularizer/Square/ReadVariableOp!^encoder_2/BiasAdd/ReadVariableOp ^encoder_2/MatMul/ReadVariableOp3^encoder_2/kernel/Regularizer/Square/ReadVariableOp!^encoder_3/BiasAdd/ReadVariableOp ^encoder_3/MatMul/ReadVariableOp3^encoder_3/kernel/Regularizer/Square/ReadVariableOp$^reg_norm_1/batchnorm/ReadVariableOp&^reg_norm_1/batchnorm/ReadVariableOp_1&^reg_norm_1/batchnorm/ReadVariableOp_2(^reg_norm_1/batchnorm/mul/ReadVariableOp!^regress_1/BiasAdd/ReadVariableOp ^regress_1/MatMul/ReadVariableOp3^regress_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2P
&ebatch_norm_0/batchnorm/ReadVariableOp&ebatch_norm_0/batchnorm/ReadVariableOp2T
(ebatch_norm_0/batchnorm/ReadVariableOp_1(ebatch_norm_0/batchnorm/ReadVariableOp_12T
(ebatch_norm_0/batchnorm/ReadVariableOp_2(ebatch_norm_0/batchnorm/ReadVariableOp_22X
*ebatch_norm_0/batchnorm/mul/ReadVariableOp*ebatch_norm_0/batchnorm/mul/ReadVariableOp2P
&ebatch_norm_1/batchnorm/ReadVariableOp&ebatch_norm_1/batchnorm/ReadVariableOp2T
(ebatch_norm_1/batchnorm/ReadVariableOp_1(ebatch_norm_1/batchnorm/ReadVariableOp_12T
(ebatch_norm_1/batchnorm/ReadVariableOp_2(ebatch_norm_1/batchnorm/ReadVariableOp_22X
*ebatch_norm_1/batchnorm/mul/ReadVariableOp*ebatch_norm_1/batchnorm/mul/ReadVariableOp2P
&ebatch_norm_2/batchnorm/ReadVariableOp&ebatch_norm_2/batchnorm/ReadVariableOp2T
(ebatch_norm_2/batchnorm/ReadVariableOp_1(ebatch_norm_2/batchnorm/ReadVariableOp_12T
(ebatch_norm_2/batchnorm/ReadVariableOp_2(ebatch_norm_2/batchnorm/ReadVariableOp_22X
*ebatch_norm_2/batchnorm/mul/ReadVariableOp*ebatch_norm_2/batchnorm/mul/ReadVariableOp2P
&ebatch_norm_3/batchnorm/ReadVariableOp&ebatch_norm_3/batchnorm/ReadVariableOp2T
(ebatch_norm_3/batchnorm/ReadVariableOp_1(ebatch_norm_3/batchnorm/ReadVariableOp_12T
(ebatch_norm_3/batchnorm/ReadVariableOp_2(ebatch_norm_3/batchnorm/ReadVariableOp_22X
*ebatch_norm_3/batchnorm/mul/ReadVariableOp*ebatch_norm_3/batchnorm/mul/ReadVariableOp2H
"ebottleneck/BiasAdd/ReadVariableOp"ebottleneck/BiasAdd/ReadVariableOp2F
!ebottleneck/MatMul/ReadVariableOp!ebottleneck/MatMul/ReadVariableOp2:
elnorm_0/add/ReadVariableOpelnorm_0/add/ReadVariableOp2>
elnorm_0/mul_2/ReadVariableOpelnorm_0/mul_2/ReadVariableOp2:
elnorm_1/add/ReadVariableOpelnorm_1/add/ReadVariableOp2>
elnorm_1/mul_2/ReadVariableOpelnorm_1/mul_2/ReadVariableOp2:
elnorm_2/add/ReadVariableOpelnorm_2/add/ReadVariableOp2>
elnorm_2/mul_2/ReadVariableOpelnorm_2/mul_2/ReadVariableOp2:
elnorm_3/add/ReadVariableOpelnorm_3/add/ReadVariableOp2>
elnorm_3/mul_2/ReadVariableOpelnorm_3/mul_2/ReadVariableOp2D
 encoder_0/BiasAdd/ReadVariableOp encoder_0/BiasAdd/ReadVariableOp2B
encoder_0/MatMul/ReadVariableOpencoder_0/MatMul/ReadVariableOp2h
2encoder_0/kernel/Regularizer/Square/ReadVariableOp2encoder_0/kernel/Regularizer/Square/ReadVariableOp2D
 encoder_1/BiasAdd/ReadVariableOp encoder_1/BiasAdd/ReadVariableOp2B
encoder_1/MatMul/ReadVariableOpencoder_1/MatMul/ReadVariableOp2h
2encoder_1/kernel/Regularizer/Square/ReadVariableOp2encoder_1/kernel/Regularizer/Square/ReadVariableOp2D
 encoder_2/BiasAdd/ReadVariableOp encoder_2/BiasAdd/ReadVariableOp2B
encoder_2/MatMul/ReadVariableOpencoder_2/MatMul/ReadVariableOp2h
2encoder_2/kernel/Regularizer/Square/ReadVariableOp2encoder_2/kernel/Regularizer/Square/ReadVariableOp2D
 encoder_3/BiasAdd/ReadVariableOp encoder_3/BiasAdd/ReadVariableOp2B
encoder_3/MatMul/ReadVariableOpencoder_3/MatMul/ReadVariableOp2h
2encoder_3/kernel/Regularizer/Square/ReadVariableOp2encoder_3/kernel/Regularizer/Square/ReadVariableOp2J
#reg_norm_1/batchnorm/ReadVariableOp#reg_norm_1/batchnorm/ReadVariableOp2N
%reg_norm_1/batchnorm/ReadVariableOp_1%reg_norm_1/batchnorm/ReadVariableOp_12N
%reg_norm_1/batchnorm/ReadVariableOp_2%reg_norm_1/batchnorm/ReadVariableOp_22R
'reg_norm_1/batchnorm/mul/ReadVariableOp'reg_norm_1/batchnorm/mul/ReadVariableOp2D
 regress_1/BiasAdd/ReadVariableOp regress_1/BiasAdd/ReadVariableOp2B
regress_1/MatMul/ReadVariableOpregress_1/MatMul/ReadVariableOp2h
2regress_1/kernel/Regularizer/Square/ReadVariableOp2regress_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_encoder_2_layer_call_fn_1360857

inputs
unknown:PP
	unknown_0:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_2_layer_call_and_return_conditional_losses_1358360o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�&
D__inference_model_1_layer_call_and_return_conditional_losses_1360536

inputs:
(encoder_0_matmul_readvariableop_resource:P7
)encoder_0_biasadd_readvariableop_resource:PC
5ebatch_norm_0_assignmovingavg_readvariableop_resource:PE
7ebatch_norm_0_assignmovingavg_1_readvariableop_resource:PA
3ebatch_norm_0_batchnorm_mul_readvariableop_resource:P=
/ebatch_norm_0_batchnorm_readvariableop_resource:P4
&elnorm_0_mul_2_readvariableop_resource:P2
$elnorm_0_add_readvariableop_resource:P:
(encoder_1_matmul_readvariableop_resource:PP7
)encoder_1_biasadd_readvariableop_resource:PC
5ebatch_norm_1_assignmovingavg_readvariableop_resource:PE
7ebatch_norm_1_assignmovingavg_1_readvariableop_resource:PA
3ebatch_norm_1_batchnorm_mul_readvariableop_resource:P=
/ebatch_norm_1_batchnorm_readvariableop_resource:P4
&elnorm_1_mul_2_readvariableop_resource:P2
$elnorm_1_add_readvariableop_resource:P:
(encoder_2_matmul_readvariableop_resource:PP7
)encoder_2_biasadd_readvariableop_resource:PC
5ebatch_norm_2_assignmovingavg_readvariableop_resource:PE
7ebatch_norm_2_assignmovingavg_1_readvariableop_resource:PA
3ebatch_norm_2_batchnorm_mul_readvariableop_resource:P=
/ebatch_norm_2_batchnorm_readvariableop_resource:P4
&elnorm_2_mul_2_readvariableop_resource:P2
$elnorm_2_add_readvariableop_resource:P:
(encoder_3_matmul_readvariableop_resource:PP7
)encoder_3_biasadd_readvariableop_resource:PC
5ebatch_norm_3_assignmovingavg_readvariableop_resource:PE
7ebatch_norm_3_assignmovingavg_1_readvariableop_resource:PA
3ebatch_norm_3_batchnorm_mul_readvariableop_resource:P=
/ebatch_norm_3_batchnorm_readvariableop_resource:P4
&elnorm_3_mul_2_readvariableop_resource:P2
$elnorm_3_add_readvariableop_resource:P<
*ebottleneck_matmul_readvariableop_resource:P9
+ebottleneck_biasadd_readvariableop_resource::
(regress_1_matmul_readvariableop_resource:27
)regress_1_biasadd_readvariableop_resource:2@
2reg_norm_1_assignmovingavg_readvariableop_resource:2B
4reg_norm_1_assignmovingavg_1_readvariableop_resource:2>
0reg_norm_1_batchnorm_mul_readvariableop_resource:2:
,reg_norm_1_batchnorm_readvariableop_resource:26
$dense_matmul_readvariableop_resource:23
%dense_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�ebatch_norm_0/AssignMovingAvg�,ebatch_norm_0/AssignMovingAvg/ReadVariableOp�ebatch_norm_0/AssignMovingAvg_1�.ebatch_norm_0/AssignMovingAvg_1/ReadVariableOp�&ebatch_norm_0/batchnorm/ReadVariableOp�*ebatch_norm_0/batchnorm/mul/ReadVariableOp�ebatch_norm_1/AssignMovingAvg�,ebatch_norm_1/AssignMovingAvg/ReadVariableOp�ebatch_norm_1/AssignMovingAvg_1�.ebatch_norm_1/AssignMovingAvg_1/ReadVariableOp�&ebatch_norm_1/batchnorm/ReadVariableOp�*ebatch_norm_1/batchnorm/mul/ReadVariableOp�ebatch_norm_2/AssignMovingAvg�,ebatch_norm_2/AssignMovingAvg/ReadVariableOp�ebatch_norm_2/AssignMovingAvg_1�.ebatch_norm_2/AssignMovingAvg_1/ReadVariableOp�&ebatch_norm_2/batchnorm/ReadVariableOp�*ebatch_norm_2/batchnorm/mul/ReadVariableOp�ebatch_norm_3/AssignMovingAvg�,ebatch_norm_3/AssignMovingAvg/ReadVariableOp�ebatch_norm_3/AssignMovingAvg_1�.ebatch_norm_3/AssignMovingAvg_1/ReadVariableOp�&ebatch_norm_3/batchnorm/ReadVariableOp�*ebatch_norm_3/batchnorm/mul/ReadVariableOp�"ebottleneck/BiasAdd/ReadVariableOp�!ebottleneck/MatMul/ReadVariableOp�elnorm_0/add/ReadVariableOp�elnorm_0/mul_2/ReadVariableOp�elnorm_1/add/ReadVariableOp�elnorm_1/mul_2/ReadVariableOp�elnorm_2/add/ReadVariableOp�elnorm_2/mul_2/ReadVariableOp�elnorm_3/add/ReadVariableOp�elnorm_3/mul_2/ReadVariableOp� encoder_0/BiasAdd/ReadVariableOp�encoder_0/MatMul/ReadVariableOp�2encoder_0/kernel/Regularizer/Square/ReadVariableOp� encoder_1/BiasAdd/ReadVariableOp�encoder_1/MatMul/ReadVariableOp�2encoder_1/kernel/Regularizer/Square/ReadVariableOp� encoder_2/BiasAdd/ReadVariableOp�encoder_2/MatMul/ReadVariableOp�2encoder_2/kernel/Regularizer/Square/ReadVariableOp� encoder_3/BiasAdd/ReadVariableOp�encoder_3/MatMul/ReadVariableOp�2encoder_3/kernel/Regularizer/Square/ReadVariableOp�reg_norm_1/AssignMovingAvg�)reg_norm_1/AssignMovingAvg/ReadVariableOp�reg_norm_1/AssignMovingAvg_1�+reg_norm_1/AssignMovingAvg_1/ReadVariableOp�#reg_norm_1/batchnorm/ReadVariableOp�'reg_norm_1/batchnorm/mul/ReadVariableOp� regress_1/BiasAdd/ReadVariableOp�regress_1/MatMul/ReadVariableOp�2regress_1/kernel/Regularizer/Square/ReadVariableOp�
encoder_0/MatMul/ReadVariableOpReadVariableOp(encoder_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0}
encoder_0/MatMulMatMulinputs'encoder_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
 encoder_0/BiasAdd/ReadVariableOpReadVariableOp)encoder_0_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
encoder_0/BiasAddBiasAddencoder_0/MatMul:product:0(encoder_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pv
,ebatch_norm_0/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
ebatch_norm_0/moments/meanMeanencoder_0/BiasAdd:output:05ebatch_norm_0/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(�
"ebatch_norm_0/moments/StopGradientStopGradient#ebatch_norm_0/moments/mean:output:0*
T0*
_output_shapes

:P�
'ebatch_norm_0/moments/SquaredDifferenceSquaredDifferenceencoder_0/BiasAdd:output:0+ebatch_norm_0/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pz
0ebatch_norm_0/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
ebatch_norm_0/moments/varianceMean+ebatch_norm_0/moments/SquaredDifference:z:09ebatch_norm_0/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(�
ebatch_norm_0/moments/SqueezeSqueeze#ebatch_norm_0/moments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 �
ebatch_norm_0/moments/Squeeze_1Squeeze'ebatch_norm_0/moments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 h
#ebatch_norm_0/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
,ebatch_norm_0/AssignMovingAvg/ReadVariableOpReadVariableOp5ebatch_norm_0_assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0�
!ebatch_norm_0/AssignMovingAvg/subSub4ebatch_norm_0/AssignMovingAvg/ReadVariableOp:value:0&ebatch_norm_0/moments/Squeeze:output:0*
T0*
_output_shapes
:P�
!ebatch_norm_0/AssignMovingAvg/mulMul%ebatch_norm_0/AssignMovingAvg/sub:z:0,ebatch_norm_0/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
ebatch_norm_0/AssignMovingAvgAssignSubVariableOp5ebatch_norm_0_assignmovingavg_readvariableop_resource%ebatch_norm_0/AssignMovingAvg/mul:z:0-^ebatch_norm_0/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0j
%ebatch_norm_0/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
.ebatch_norm_0/AssignMovingAvg_1/ReadVariableOpReadVariableOp7ebatch_norm_0_assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
#ebatch_norm_0/AssignMovingAvg_1/subSub6ebatch_norm_0/AssignMovingAvg_1/ReadVariableOp:value:0(ebatch_norm_0/moments/Squeeze_1:output:0*
T0*
_output_shapes
:P�
#ebatch_norm_0/AssignMovingAvg_1/mulMul'ebatch_norm_0/AssignMovingAvg_1/sub:z:0.ebatch_norm_0/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
ebatch_norm_0/AssignMovingAvg_1AssignSubVariableOp7ebatch_norm_0_assignmovingavg_1_readvariableop_resource'ebatch_norm_0/AssignMovingAvg_1/mul:z:0/^ebatch_norm_0/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0b
ebatch_norm_0/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
ebatch_norm_0/batchnorm/addAddV2(ebatch_norm_0/moments/Squeeze_1:output:0&ebatch_norm_0/batchnorm/add/y:output:0*
T0*
_output_shapes
:Pl
ebatch_norm_0/batchnorm/RsqrtRsqrtebatch_norm_0/batchnorm/add:z:0*
T0*
_output_shapes
:P�
*ebatch_norm_0/batchnorm/mul/ReadVariableOpReadVariableOp3ebatch_norm_0_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_0/batchnorm/mulMul!ebatch_norm_0/batchnorm/Rsqrt:y:02ebatch_norm_0/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
ebatch_norm_0/batchnorm/mul_1Mulencoder_0/BiasAdd:output:0ebatch_norm_0/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
ebatch_norm_0/batchnorm/mul_2Mul&ebatch_norm_0/moments/Squeeze:output:0ebatch_norm_0/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
&ebatch_norm_0/batchnorm/ReadVariableOpReadVariableOp/ebatch_norm_0_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_0/batchnorm/subSub.ebatch_norm_0/batchnorm/ReadVariableOp:value:0!ebatch_norm_0/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
ebatch_norm_0/batchnorm/add_1AddV2!ebatch_norm_0/batchnorm/mul_1:z:0ebatch_norm_0/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P_
elnorm_0/ShapeShape!ebatch_norm_0/batchnorm/add_1:z:0*
T0*
_output_shapes
:f
elnorm_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
elnorm_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
elnorm_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_0/strided_sliceStridedSliceelnorm_0/Shape:output:0%elnorm_0/strided_slice/stack:output:0'elnorm_0/strided_slice/stack_1:output:0'elnorm_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
elnorm_0/mul/xConst*
_output_shapes
: *
dtype0*
value	B :n
elnorm_0/mulMulelnorm_0/mul/x:output:0elnorm_0/strided_slice:output:0*
T0*
_output_shapes
: h
elnorm_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_0/strided_slice_1StridedSliceelnorm_0/Shape:output:0'elnorm_0/strided_slice_1/stack:output:0)elnorm_0/strided_slice_1/stack_1:output:0)elnorm_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
elnorm_0/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :t
elnorm_0/mul_1Mulelnorm_0/mul_1/x:output:0!elnorm_0/strided_slice_1:output:0*
T0*
_output_shapes
: Z
elnorm_0/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Z
elnorm_0/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
elnorm_0/Reshape/shapePack!elnorm_0/Reshape/shape/0:output:0elnorm_0/mul:z:0elnorm_0/mul_1:z:0!elnorm_0/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
elnorm_0/ReshapeReshape!ebatch_norm_0/batchnorm/add_1:z:0elnorm_0/Reshape/shape:output:0*
T0*/
_output_shapes
:���������P\
elnorm_0/ones/packedPackelnorm_0/mul:z:0*
N*
T0*
_output_shapes
:X
elnorm_0/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
elnorm_0/onesFillelnorm_0/ones/packed:output:0elnorm_0/ones/Const:output:0*
T0*#
_output_shapes
:���������]
elnorm_0/zeros/packedPackelnorm_0/mul:z:0*
N*
T0*
_output_shapes
:Y
elnorm_0/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
elnorm_0/zerosFillelnorm_0/zeros/packed:output:0elnorm_0/zeros/Const:output:0*
T0*#
_output_shapes
:���������Q
elnorm_0/ConstConst*
_output_shapes
: *
dtype0*
valueB S
elnorm_0/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
elnorm_0/FusedBatchNormV3FusedBatchNormV3elnorm_0/Reshape:output:0elnorm_0/ones:output:0elnorm_0/zeros:output:0elnorm_0/Const:output:0elnorm_0/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
elnorm_0/Reshape_1Reshapeelnorm_0/FusedBatchNormV3:y:0elnorm_0/Shape:output:0*
T0*'
_output_shapes
:���������P�
elnorm_0/mul_2/ReadVariableOpReadVariableOp&elnorm_0_mul_2_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_0/mul_2Mulelnorm_0/Reshape_1:output:0%elnorm_0/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P|
elnorm_0/add/ReadVariableOpReadVariableOp$elnorm_0_add_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_0/addAddV2elnorm_0/mul_2:z:0#elnorm_0/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
encoder_1/MatMul/ReadVariableOpReadVariableOp(encoder_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
encoder_1/MatMulMatMulelnorm_0/add:z:0'encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
 encoder_1/BiasAdd/ReadVariableOpReadVariableOp)encoder_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
encoder_1/BiasAddBiasAddencoder_1/MatMul:product:0(encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pv
,ebatch_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
ebatch_norm_1/moments/meanMeanencoder_1/BiasAdd:output:05ebatch_norm_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(�
"ebatch_norm_1/moments/StopGradientStopGradient#ebatch_norm_1/moments/mean:output:0*
T0*
_output_shapes

:P�
'ebatch_norm_1/moments/SquaredDifferenceSquaredDifferenceencoder_1/BiasAdd:output:0+ebatch_norm_1/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pz
0ebatch_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
ebatch_norm_1/moments/varianceMean+ebatch_norm_1/moments/SquaredDifference:z:09ebatch_norm_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(�
ebatch_norm_1/moments/SqueezeSqueeze#ebatch_norm_1/moments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 �
ebatch_norm_1/moments/Squeeze_1Squeeze'ebatch_norm_1/moments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 h
#ebatch_norm_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
,ebatch_norm_1/AssignMovingAvg/ReadVariableOpReadVariableOp5ebatch_norm_1_assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0�
!ebatch_norm_1/AssignMovingAvg/subSub4ebatch_norm_1/AssignMovingAvg/ReadVariableOp:value:0&ebatch_norm_1/moments/Squeeze:output:0*
T0*
_output_shapes
:P�
!ebatch_norm_1/AssignMovingAvg/mulMul%ebatch_norm_1/AssignMovingAvg/sub:z:0,ebatch_norm_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
ebatch_norm_1/AssignMovingAvgAssignSubVariableOp5ebatch_norm_1_assignmovingavg_readvariableop_resource%ebatch_norm_1/AssignMovingAvg/mul:z:0-^ebatch_norm_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0j
%ebatch_norm_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
.ebatch_norm_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp7ebatch_norm_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
#ebatch_norm_1/AssignMovingAvg_1/subSub6ebatch_norm_1/AssignMovingAvg_1/ReadVariableOp:value:0(ebatch_norm_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:P�
#ebatch_norm_1/AssignMovingAvg_1/mulMul'ebatch_norm_1/AssignMovingAvg_1/sub:z:0.ebatch_norm_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
ebatch_norm_1/AssignMovingAvg_1AssignSubVariableOp7ebatch_norm_1_assignmovingavg_1_readvariableop_resource'ebatch_norm_1/AssignMovingAvg_1/mul:z:0/^ebatch_norm_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0b
ebatch_norm_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
ebatch_norm_1/batchnorm/addAddV2(ebatch_norm_1/moments/Squeeze_1:output:0&ebatch_norm_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:Pl
ebatch_norm_1/batchnorm/RsqrtRsqrtebatch_norm_1/batchnorm/add:z:0*
T0*
_output_shapes
:P�
*ebatch_norm_1/batchnorm/mul/ReadVariableOpReadVariableOp3ebatch_norm_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_1/batchnorm/mulMul!ebatch_norm_1/batchnorm/Rsqrt:y:02ebatch_norm_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
ebatch_norm_1/batchnorm/mul_1Mulencoder_1/BiasAdd:output:0ebatch_norm_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
ebatch_norm_1/batchnorm/mul_2Mul&ebatch_norm_1/moments/Squeeze:output:0ebatch_norm_1/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
&ebatch_norm_1/batchnorm/ReadVariableOpReadVariableOp/ebatch_norm_1_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_1/batchnorm/subSub.ebatch_norm_1/batchnorm/ReadVariableOp:value:0!ebatch_norm_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
ebatch_norm_1/batchnorm/add_1AddV2!ebatch_norm_1/batchnorm/mul_1:z:0ebatch_norm_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P_
elnorm_1/ShapeShape!ebatch_norm_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:f
elnorm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
elnorm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
elnorm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_1/strided_sliceStridedSliceelnorm_1/Shape:output:0%elnorm_1/strided_slice/stack:output:0'elnorm_1/strided_slice/stack_1:output:0'elnorm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
elnorm_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :n
elnorm_1/mulMulelnorm_1/mul/x:output:0elnorm_1/strided_slice:output:0*
T0*
_output_shapes
: h
elnorm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_1/strided_slice_1StridedSliceelnorm_1/Shape:output:0'elnorm_1/strided_slice_1/stack:output:0)elnorm_1/strided_slice_1/stack_1:output:0)elnorm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
elnorm_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :t
elnorm_1/mul_1Mulelnorm_1/mul_1/x:output:0!elnorm_1/strided_slice_1:output:0*
T0*
_output_shapes
: Z
elnorm_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Z
elnorm_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
elnorm_1/Reshape/shapePack!elnorm_1/Reshape/shape/0:output:0elnorm_1/mul:z:0elnorm_1/mul_1:z:0!elnorm_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
elnorm_1/ReshapeReshape!ebatch_norm_1/batchnorm/add_1:z:0elnorm_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������P\
elnorm_1/ones/packedPackelnorm_1/mul:z:0*
N*
T0*
_output_shapes
:X
elnorm_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
elnorm_1/onesFillelnorm_1/ones/packed:output:0elnorm_1/ones/Const:output:0*
T0*#
_output_shapes
:���������]
elnorm_1/zeros/packedPackelnorm_1/mul:z:0*
N*
T0*
_output_shapes
:Y
elnorm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
elnorm_1/zerosFillelnorm_1/zeros/packed:output:0elnorm_1/zeros/Const:output:0*
T0*#
_output_shapes
:���������Q
elnorm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB S
elnorm_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
elnorm_1/FusedBatchNormV3FusedBatchNormV3elnorm_1/Reshape:output:0elnorm_1/ones:output:0elnorm_1/zeros:output:0elnorm_1/Const:output:0elnorm_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
elnorm_1/Reshape_1Reshapeelnorm_1/FusedBatchNormV3:y:0elnorm_1/Shape:output:0*
T0*'
_output_shapes
:���������P�
elnorm_1/mul_2/ReadVariableOpReadVariableOp&elnorm_1_mul_2_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_1/mul_2Mulelnorm_1/Reshape_1:output:0%elnorm_1/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P|
elnorm_1/add/ReadVariableOpReadVariableOp$elnorm_1_add_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_1/addAddV2elnorm_1/mul_2:z:0#elnorm_1/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
encoder_2/MatMul/ReadVariableOpReadVariableOp(encoder_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
encoder_2/MatMulMatMulelnorm_1/add:z:0'encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
 encoder_2/BiasAdd/ReadVariableOpReadVariableOp)encoder_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
encoder_2/BiasAddBiasAddencoder_2/MatMul:product:0(encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pv
,ebatch_norm_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
ebatch_norm_2/moments/meanMeanencoder_2/BiasAdd:output:05ebatch_norm_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(�
"ebatch_norm_2/moments/StopGradientStopGradient#ebatch_norm_2/moments/mean:output:0*
T0*
_output_shapes

:P�
'ebatch_norm_2/moments/SquaredDifferenceSquaredDifferenceencoder_2/BiasAdd:output:0+ebatch_norm_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pz
0ebatch_norm_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
ebatch_norm_2/moments/varianceMean+ebatch_norm_2/moments/SquaredDifference:z:09ebatch_norm_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(�
ebatch_norm_2/moments/SqueezeSqueeze#ebatch_norm_2/moments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 �
ebatch_norm_2/moments/Squeeze_1Squeeze'ebatch_norm_2/moments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 h
#ebatch_norm_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
,ebatch_norm_2/AssignMovingAvg/ReadVariableOpReadVariableOp5ebatch_norm_2_assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0�
!ebatch_norm_2/AssignMovingAvg/subSub4ebatch_norm_2/AssignMovingAvg/ReadVariableOp:value:0&ebatch_norm_2/moments/Squeeze:output:0*
T0*
_output_shapes
:P�
!ebatch_norm_2/AssignMovingAvg/mulMul%ebatch_norm_2/AssignMovingAvg/sub:z:0,ebatch_norm_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
ebatch_norm_2/AssignMovingAvgAssignSubVariableOp5ebatch_norm_2_assignmovingavg_readvariableop_resource%ebatch_norm_2/AssignMovingAvg/mul:z:0-^ebatch_norm_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0j
%ebatch_norm_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
.ebatch_norm_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp7ebatch_norm_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
#ebatch_norm_2/AssignMovingAvg_1/subSub6ebatch_norm_2/AssignMovingAvg_1/ReadVariableOp:value:0(ebatch_norm_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:P�
#ebatch_norm_2/AssignMovingAvg_1/mulMul'ebatch_norm_2/AssignMovingAvg_1/sub:z:0.ebatch_norm_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
ebatch_norm_2/AssignMovingAvg_1AssignSubVariableOp7ebatch_norm_2_assignmovingavg_1_readvariableop_resource'ebatch_norm_2/AssignMovingAvg_1/mul:z:0/^ebatch_norm_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0b
ebatch_norm_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
ebatch_norm_2/batchnorm/addAddV2(ebatch_norm_2/moments/Squeeze_1:output:0&ebatch_norm_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:Pl
ebatch_norm_2/batchnorm/RsqrtRsqrtebatch_norm_2/batchnorm/add:z:0*
T0*
_output_shapes
:P�
*ebatch_norm_2/batchnorm/mul/ReadVariableOpReadVariableOp3ebatch_norm_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_2/batchnorm/mulMul!ebatch_norm_2/batchnorm/Rsqrt:y:02ebatch_norm_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
ebatch_norm_2/batchnorm/mul_1Mulencoder_2/BiasAdd:output:0ebatch_norm_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
ebatch_norm_2/batchnorm/mul_2Mul&ebatch_norm_2/moments/Squeeze:output:0ebatch_norm_2/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
&ebatch_norm_2/batchnorm/ReadVariableOpReadVariableOp/ebatch_norm_2_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_2/batchnorm/subSub.ebatch_norm_2/batchnorm/ReadVariableOp:value:0!ebatch_norm_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
ebatch_norm_2/batchnorm/add_1AddV2!ebatch_norm_2/batchnorm/mul_1:z:0ebatch_norm_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P_
elnorm_2/ShapeShape!ebatch_norm_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:f
elnorm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
elnorm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
elnorm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_2/strided_sliceStridedSliceelnorm_2/Shape:output:0%elnorm_2/strided_slice/stack:output:0'elnorm_2/strided_slice/stack_1:output:0'elnorm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
elnorm_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :n
elnorm_2/mulMulelnorm_2/mul/x:output:0elnorm_2/strided_slice:output:0*
T0*
_output_shapes
: h
elnorm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_2/strided_slice_1StridedSliceelnorm_2/Shape:output:0'elnorm_2/strided_slice_1/stack:output:0)elnorm_2/strided_slice_1/stack_1:output:0)elnorm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
elnorm_2/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :t
elnorm_2/mul_1Mulelnorm_2/mul_1/x:output:0!elnorm_2/strided_slice_1:output:0*
T0*
_output_shapes
: Z
elnorm_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Z
elnorm_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
elnorm_2/Reshape/shapePack!elnorm_2/Reshape/shape/0:output:0elnorm_2/mul:z:0elnorm_2/mul_1:z:0!elnorm_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
elnorm_2/ReshapeReshape!ebatch_norm_2/batchnorm/add_1:z:0elnorm_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������P\
elnorm_2/ones/packedPackelnorm_2/mul:z:0*
N*
T0*
_output_shapes
:X
elnorm_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
elnorm_2/onesFillelnorm_2/ones/packed:output:0elnorm_2/ones/Const:output:0*
T0*#
_output_shapes
:���������]
elnorm_2/zeros/packedPackelnorm_2/mul:z:0*
N*
T0*
_output_shapes
:Y
elnorm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
elnorm_2/zerosFillelnorm_2/zeros/packed:output:0elnorm_2/zeros/Const:output:0*
T0*#
_output_shapes
:���������Q
elnorm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB S
elnorm_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
elnorm_2/FusedBatchNormV3FusedBatchNormV3elnorm_2/Reshape:output:0elnorm_2/ones:output:0elnorm_2/zeros:output:0elnorm_2/Const:output:0elnorm_2/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
elnorm_2/Reshape_1Reshapeelnorm_2/FusedBatchNormV3:y:0elnorm_2/Shape:output:0*
T0*'
_output_shapes
:���������P�
elnorm_2/mul_2/ReadVariableOpReadVariableOp&elnorm_2_mul_2_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_2/mul_2Mulelnorm_2/Reshape_1:output:0%elnorm_2/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P|
elnorm_2/add/ReadVariableOpReadVariableOp$elnorm_2_add_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_2/addAddV2elnorm_2/mul_2:z:0#elnorm_2/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
encoder_3/MatMul/ReadVariableOpReadVariableOp(encoder_3_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
encoder_3/MatMulMatMulelnorm_2/add:z:0'encoder_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
 encoder_3/BiasAdd/ReadVariableOpReadVariableOp)encoder_3_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
encoder_3/BiasAddBiasAddencoder_3/MatMul:product:0(encoder_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pv
,ebatch_norm_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
ebatch_norm_3/moments/meanMeanencoder_3/BiasAdd:output:05ebatch_norm_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(�
"ebatch_norm_3/moments/StopGradientStopGradient#ebatch_norm_3/moments/mean:output:0*
T0*
_output_shapes

:P�
'ebatch_norm_3/moments/SquaredDifferenceSquaredDifferenceencoder_3/BiasAdd:output:0+ebatch_norm_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pz
0ebatch_norm_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
ebatch_norm_3/moments/varianceMean+ebatch_norm_3/moments/SquaredDifference:z:09ebatch_norm_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(�
ebatch_norm_3/moments/SqueezeSqueeze#ebatch_norm_3/moments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 �
ebatch_norm_3/moments/Squeeze_1Squeeze'ebatch_norm_3/moments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 h
#ebatch_norm_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
,ebatch_norm_3/AssignMovingAvg/ReadVariableOpReadVariableOp5ebatch_norm_3_assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0�
!ebatch_norm_3/AssignMovingAvg/subSub4ebatch_norm_3/AssignMovingAvg/ReadVariableOp:value:0&ebatch_norm_3/moments/Squeeze:output:0*
T0*
_output_shapes
:P�
!ebatch_norm_3/AssignMovingAvg/mulMul%ebatch_norm_3/AssignMovingAvg/sub:z:0,ebatch_norm_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
ebatch_norm_3/AssignMovingAvgAssignSubVariableOp5ebatch_norm_3_assignmovingavg_readvariableop_resource%ebatch_norm_3/AssignMovingAvg/mul:z:0-^ebatch_norm_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0j
%ebatch_norm_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
.ebatch_norm_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp7ebatch_norm_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
#ebatch_norm_3/AssignMovingAvg_1/subSub6ebatch_norm_3/AssignMovingAvg_1/ReadVariableOp:value:0(ebatch_norm_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:P�
#ebatch_norm_3/AssignMovingAvg_1/mulMul'ebatch_norm_3/AssignMovingAvg_1/sub:z:0.ebatch_norm_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
ebatch_norm_3/AssignMovingAvg_1AssignSubVariableOp7ebatch_norm_3_assignmovingavg_1_readvariableop_resource'ebatch_norm_3/AssignMovingAvg_1/mul:z:0/^ebatch_norm_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0b
ebatch_norm_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
ebatch_norm_3/batchnorm/addAddV2(ebatch_norm_3/moments/Squeeze_1:output:0&ebatch_norm_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:Pl
ebatch_norm_3/batchnorm/RsqrtRsqrtebatch_norm_3/batchnorm/add:z:0*
T0*
_output_shapes
:P�
*ebatch_norm_3/batchnorm/mul/ReadVariableOpReadVariableOp3ebatch_norm_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_3/batchnorm/mulMul!ebatch_norm_3/batchnorm/Rsqrt:y:02ebatch_norm_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
ebatch_norm_3/batchnorm/mul_1Mulencoder_3/BiasAdd:output:0ebatch_norm_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
ebatch_norm_3/batchnorm/mul_2Mul&ebatch_norm_3/moments/Squeeze:output:0ebatch_norm_3/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
&ebatch_norm_3/batchnorm/ReadVariableOpReadVariableOp/ebatch_norm_3_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0�
ebatch_norm_3/batchnorm/subSub.ebatch_norm_3/batchnorm/ReadVariableOp:value:0!ebatch_norm_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
ebatch_norm_3/batchnorm/add_1AddV2!ebatch_norm_3/batchnorm/mul_1:z:0ebatch_norm_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P_
elnorm_3/ShapeShape!ebatch_norm_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:f
elnorm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
elnorm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
elnorm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_3/strided_sliceStridedSliceelnorm_3/Shape:output:0%elnorm_3/strided_slice/stack:output:0'elnorm_3/strided_slice/stack_1:output:0'elnorm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
elnorm_3/mul/xConst*
_output_shapes
: *
dtype0*
value	B :n
elnorm_3/mulMulelnorm_3/mul/x:output:0elnorm_3/strided_slice:output:0*
T0*
_output_shapes
: h
elnorm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 elnorm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
elnorm_3/strided_slice_1StridedSliceelnorm_3/Shape:output:0'elnorm_3/strided_slice_1/stack:output:0)elnorm_3/strided_slice_1/stack_1:output:0)elnorm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
elnorm_3/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :t
elnorm_3/mul_1Mulelnorm_3/mul_1/x:output:0!elnorm_3/strided_slice_1:output:0*
T0*
_output_shapes
: Z
elnorm_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Z
elnorm_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
elnorm_3/Reshape/shapePack!elnorm_3/Reshape/shape/0:output:0elnorm_3/mul:z:0elnorm_3/mul_1:z:0!elnorm_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
elnorm_3/ReshapeReshape!ebatch_norm_3/batchnorm/add_1:z:0elnorm_3/Reshape/shape:output:0*
T0*/
_output_shapes
:���������P\
elnorm_3/ones/packedPackelnorm_3/mul:z:0*
N*
T0*
_output_shapes
:X
elnorm_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
elnorm_3/onesFillelnorm_3/ones/packed:output:0elnorm_3/ones/Const:output:0*
T0*#
_output_shapes
:���������]
elnorm_3/zeros/packedPackelnorm_3/mul:z:0*
N*
T0*
_output_shapes
:Y
elnorm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
elnorm_3/zerosFillelnorm_3/zeros/packed:output:0elnorm_3/zeros/Const:output:0*
T0*#
_output_shapes
:���������Q
elnorm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB S
elnorm_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
elnorm_3/FusedBatchNormV3FusedBatchNormV3elnorm_3/Reshape:output:0elnorm_3/ones:output:0elnorm_3/zeros:output:0elnorm_3/Const:output:0elnorm_3/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
elnorm_3/Reshape_1Reshapeelnorm_3/FusedBatchNormV3:y:0elnorm_3/Shape:output:0*
T0*'
_output_shapes
:���������P�
elnorm_3/mul_2/ReadVariableOpReadVariableOp&elnorm_3_mul_2_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_3/mul_2Mulelnorm_3/Reshape_1:output:0%elnorm_3/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P|
elnorm_3/add/ReadVariableOpReadVariableOp$elnorm_3_add_readvariableop_resource*
_output_shapes
:P*
dtype0�
elnorm_3/addAddV2elnorm_3/mul_2:z:0#elnorm_3/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
!ebottleneck/MatMul/ReadVariableOpReadVariableOp*ebottleneck_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0�
ebottleneck/MatMulMatMulelnorm_3/add:z:0)ebottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"ebottleneck/BiasAdd/ReadVariableOpReadVariableOp+ebottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
ebottleneck/BiasAddBiasAddebottleneck/MatMul:product:0*ebottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
ebottleneck/ReluReluebottleneck/BiasAdd:output:0*
T0*'
_output_shapes
:���������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapeebottleneck/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:����������
regress_1/MatMul/ReadVariableOpReadVariableOp(regress_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
regress_1/MatMulMatMulflatten/Reshape:output:0'regress_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
 regress_1/BiasAdd/ReadVariableOpReadVariableOp)regress_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
regress_1/BiasAddBiasAddregress_1/MatMul:product:0(regress_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2d
regress_1/ReluReluregress_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2s
)reg_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
reg_norm_1/moments/meanMeanregress_1/Relu:activations:02reg_norm_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(z
reg_norm_1/moments/StopGradientStopGradient reg_norm_1/moments/mean:output:0*
T0*
_output_shapes

:2�
$reg_norm_1/moments/SquaredDifferenceSquaredDifferenceregress_1/Relu:activations:0(reg_norm_1/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������2w
-reg_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
reg_norm_1/moments/varianceMean(reg_norm_1/moments/SquaredDifference:z:06reg_norm_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(�
reg_norm_1/moments/SqueezeSqueeze reg_norm_1/moments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 �
reg_norm_1/moments/Squeeze_1Squeeze$reg_norm_1/moments/variance:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 e
 reg_norm_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
)reg_norm_1/AssignMovingAvg/ReadVariableOpReadVariableOp2reg_norm_1_assignmovingavg_readvariableop_resource*
_output_shapes
:2*
dtype0�
reg_norm_1/AssignMovingAvg/subSub1reg_norm_1/AssignMovingAvg/ReadVariableOp:value:0#reg_norm_1/moments/Squeeze:output:0*
T0*
_output_shapes
:2�
reg_norm_1/AssignMovingAvg/mulMul"reg_norm_1/AssignMovingAvg/sub:z:0)reg_norm_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2�
reg_norm_1/AssignMovingAvgAssignSubVariableOp2reg_norm_1_assignmovingavg_readvariableop_resource"reg_norm_1/AssignMovingAvg/mul:z:0*^reg_norm_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0g
"reg_norm_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
+reg_norm_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp4reg_norm_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:2*
dtype0�
 reg_norm_1/AssignMovingAvg_1/subSub3reg_norm_1/AssignMovingAvg_1/ReadVariableOp:value:0%reg_norm_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2�
 reg_norm_1/AssignMovingAvg_1/mulMul$reg_norm_1/AssignMovingAvg_1/sub:z:0+reg_norm_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2�
reg_norm_1/AssignMovingAvg_1AssignSubVariableOp4reg_norm_1_assignmovingavg_1_readvariableop_resource$reg_norm_1/AssignMovingAvg_1/mul:z:0,^reg_norm_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0_
reg_norm_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
reg_norm_1/batchnorm/addAddV2%reg_norm_1/moments/Squeeze_1:output:0#reg_norm_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2f
reg_norm_1/batchnorm/RsqrtRsqrtreg_norm_1/batchnorm/add:z:0*
T0*
_output_shapes
:2�
'reg_norm_1/batchnorm/mul/ReadVariableOpReadVariableOp0reg_norm_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0�
reg_norm_1/batchnorm/mulMulreg_norm_1/batchnorm/Rsqrt:y:0/reg_norm_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2�
reg_norm_1/batchnorm/mul_1Mulregress_1/Relu:activations:0reg_norm_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2�
reg_norm_1/batchnorm/mul_2Mul#reg_norm_1/moments/Squeeze:output:0reg_norm_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2�
#reg_norm_1/batchnorm/ReadVariableOpReadVariableOp,reg_norm_1_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0�
reg_norm_1/batchnorm/subSub+reg_norm_1/batchnorm/ReadVariableOp:value:0reg_norm_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2�
reg_norm_1/batchnorm/add_1AddV2reg_norm_1/batchnorm/mul_1:z:0reg_norm_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
dense/MatMulMatMulreg_norm_1/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2encoder_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(encoder_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0�
#encoder_0/kernel/Regularizer/SquareSquare:encoder_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Ps
"encoder_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_0/kernel/Regularizer/SumSum'encoder_0/kernel/Regularizer/Square:y:0+encoder_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_0/kernel/Regularizer/mulMul+encoder_0/kernel/Regularizer/mul/x:output:0)encoder_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(encoder_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_1/kernel/Regularizer/SquareSquare:encoder_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_1/kernel/Regularizer/SumSum'encoder_1/kernel/Regularizer/Square:y:0+encoder_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_1/kernel/Regularizer/mulMul+encoder_1/kernel/Regularizer/mul/x:output:0)encoder_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(encoder_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_2/kernel/Regularizer/SquareSquare:encoder_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_2/kernel/Regularizer/SumSum'encoder_2/kernel/Regularizer/Square:y:0+encoder_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_2/kernel/Regularizer/mulMul+encoder_2/kernel/Regularizer/mul/x:output:0)encoder_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(encoder_3_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_3/kernel/Regularizer/SquareSquare:encoder_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_3/kernel/Regularizer/SumSum'encoder_3/kernel/Regularizer/Square:y:0+encoder_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_3/kernel/Regularizer/mulMul+encoder_3/kernel/Regularizer/mul/x:output:0)encoder_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2regress_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(regress_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
#regress_1/kernel/Regularizer/SquareSquare:regress_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2s
"regress_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 regress_1/kernel/Regularizer/SumSum'regress_1/kernel/Regularizer/Square:y:0+regress_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"regress_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 regress_1/kernel/Regularizer/mulMul+regress_1/kernel/Regularizer/mul/x:output:0)regress_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^ebatch_norm_0/AssignMovingAvg-^ebatch_norm_0/AssignMovingAvg/ReadVariableOp ^ebatch_norm_0/AssignMovingAvg_1/^ebatch_norm_0/AssignMovingAvg_1/ReadVariableOp'^ebatch_norm_0/batchnorm/ReadVariableOp+^ebatch_norm_0/batchnorm/mul/ReadVariableOp^ebatch_norm_1/AssignMovingAvg-^ebatch_norm_1/AssignMovingAvg/ReadVariableOp ^ebatch_norm_1/AssignMovingAvg_1/^ebatch_norm_1/AssignMovingAvg_1/ReadVariableOp'^ebatch_norm_1/batchnorm/ReadVariableOp+^ebatch_norm_1/batchnorm/mul/ReadVariableOp^ebatch_norm_2/AssignMovingAvg-^ebatch_norm_2/AssignMovingAvg/ReadVariableOp ^ebatch_norm_2/AssignMovingAvg_1/^ebatch_norm_2/AssignMovingAvg_1/ReadVariableOp'^ebatch_norm_2/batchnorm/ReadVariableOp+^ebatch_norm_2/batchnorm/mul/ReadVariableOp^ebatch_norm_3/AssignMovingAvg-^ebatch_norm_3/AssignMovingAvg/ReadVariableOp ^ebatch_norm_3/AssignMovingAvg_1/^ebatch_norm_3/AssignMovingAvg_1/ReadVariableOp'^ebatch_norm_3/batchnorm/ReadVariableOp+^ebatch_norm_3/batchnorm/mul/ReadVariableOp#^ebottleneck/BiasAdd/ReadVariableOp"^ebottleneck/MatMul/ReadVariableOp^elnorm_0/add/ReadVariableOp^elnorm_0/mul_2/ReadVariableOp^elnorm_1/add/ReadVariableOp^elnorm_1/mul_2/ReadVariableOp^elnorm_2/add/ReadVariableOp^elnorm_2/mul_2/ReadVariableOp^elnorm_3/add/ReadVariableOp^elnorm_3/mul_2/ReadVariableOp!^encoder_0/BiasAdd/ReadVariableOp ^encoder_0/MatMul/ReadVariableOp3^encoder_0/kernel/Regularizer/Square/ReadVariableOp!^encoder_1/BiasAdd/ReadVariableOp ^encoder_1/MatMul/ReadVariableOp3^encoder_1/kernel/Regularizer/Square/ReadVariableOp!^encoder_2/BiasAdd/ReadVariableOp ^encoder_2/MatMul/ReadVariableOp3^encoder_2/kernel/Regularizer/Square/ReadVariableOp!^encoder_3/BiasAdd/ReadVariableOp ^encoder_3/MatMul/ReadVariableOp3^encoder_3/kernel/Regularizer/Square/ReadVariableOp^reg_norm_1/AssignMovingAvg*^reg_norm_1/AssignMovingAvg/ReadVariableOp^reg_norm_1/AssignMovingAvg_1,^reg_norm_1/AssignMovingAvg_1/ReadVariableOp$^reg_norm_1/batchnorm/ReadVariableOp(^reg_norm_1/batchnorm/mul/ReadVariableOp!^regress_1/BiasAdd/ReadVariableOp ^regress_1/MatMul/ReadVariableOp3^regress_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
ebatch_norm_0/AssignMovingAvgebatch_norm_0/AssignMovingAvg2\
,ebatch_norm_0/AssignMovingAvg/ReadVariableOp,ebatch_norm_0/AssignMovingAvg/ReadVariableOp2B
ebatch_norm_0/AssignMovingAvg_1ebatch_norm_0/AssignMovingAvg_12`
.ebatch_norm_0/AssignMovingAvg_1/ReadVariableOp.ebatch_norm_0/AssignMovingAvg_1/ReadVariableOp2P
&ebatch_norm_0/batchnorm/ReadVariableOp&ebatch_norm_0/batchnorm/ReadVariableOp2X
*ebatch_norm_0/batchnorm/mul/ReadVariableOp*ebatch_norm_0/batchnorm/mul/ReadVariableOp2>
ebatch_norm_1/AssignMovingAvgebatch_norm_1/AssignMovingAvg2\
,ebatch_norm_1/AssignMovingAvg/ReadVariableOp,ebatch_norm_1/AssignMovingAvg/ReadVariableOp2B
ebatch_norm_1/AssignMovingAvg_1ebatch_norm_1/AssignMovingAvg_12`
.ebatch_norm_1/AssignMovingAvg_1/ReadVariableOp.ebatch_norm_1/AssignMovingAvg_1/ReadVariableOp2P
&ebatch_norm_1/batchnorm/ReadVariableOp&ebatch_norm_1/batchnorm/ReadVariableOp2X
*ebatch_norm_1/batchnorm/mul/ReadVariableOp*ebatch_norm_1/batchnorm/mul/ReadVariableOp2>
ebatch_norm_2/AssignMovingAvgebatch_norm_2/AssignMovingAvg2\
,ebatch_norm_2/AssignMovingAvg/ReadVariableOp,ebatch_norm_2/AssignMovingAvg/ReadVariableOp2B
ebatch_norm_2/AssignMovingAvg_1ebatch_norm_2/AssignMovingAvg_12`
.ebatch_norm_2/AssignMovingAvg_1/ReadVariableOp.ebatch_norm_2/AssignMovingAvg_1/ReadVariableOp2P
&ebatch_norm_2/batchnorm/ReadVariableOp&ebatch_norm_2/batchnorm/ReadVariableOp2X
*ebatch_norm_2/batchnorm/mul/ReadVariableOp*ebatch_norm_2/batchnorm/mul/ReadVariableOp2>
ebatch_norm_3/AssignMovingAvgebatch_norm_3/AssignMovingAvg2\
,ebatch_norm_3/AssignMovingAvg/ReadVariableOp,ebatch_norm_3/AssignMovingAvg/ReadVariableOp2B
ebatch_norm_3/AssignMovingAvg_1ebatch_norm_3/AssignMovingAvg_12`
.ebatch_norm_3/AssignMovingAvg_1/ReadVariableOp.ebatch_norm_3/AssignMovingAvg_1/ReadVariableOp2P
&ebatch_norm_3/batchnorm/ReadVariableOp&ebatch_norm_3/batchnorm/ReadVariableOp2X
*ebatch_norm_3/batchnorm/mul/ReadVariableOp*ebatch_norm_3/batchnorm/mul/ReadVariableOp2H
"ebottleneck/BiasAdd/ReadVariableOp"ebottleneck/BiasAdd/ReadVariableOp2F
!ebottleneck/MatMul/ReadVariableOp!ebottleneck/MatMul/ReadVariableOp2:
elnorm_0/add/ReadVariableOpelnorm_0/add/ReadVariableOp2>
elnorm_0/mul_2/ReadVariableOpelnorm_0/mul_2/ReadVariableOp2:
elnorm_1/add/ReadVariableOpelnorm_1/add/ReadVariableOp2>
elnorm_1/mul_2/ReadVariableOpelnorm_1/mul_2/ReadVariableOp2:
elnorm_2/add/ReadVariableOpelnorm_2/add/ReadVariableOp2>
elnorm_2/mul_2/ReadVariableOpelnorm_2/mul_2/ReadVariableOp2:
elnorm_3/add/ReadVariableOpelnorm_3/add/ReadVariableOp2>
elnorm_3/mul_2/ReadVariableOpelnorm_3/mul_2/ReadVariableOp2D
 encoder_0/BiasAdd/ReadVariableOp encoder_0/BiasAdd/ReadVariableOp2B
encoder_0/MatMul/ReadVariableOpencoder_0/MatMul/ReadVariableOp2h
2encoder_0/kernel/Regularizer/Square/ReadVariableOp2encoder_0/kernel/Regularizer/Square/ReadVariableOp2D
 encoder_1/BiasAdd/ReadVariableOp encoder_1/BiasAdd/ReadVariableOp2B
encoder_1/MatMul/ReadVariableOpencoder_1/MatMul/ReadVariableOp2h
2encoder_1/kernel/Regularizer/Square/ReadVariableOp2encoder_1/kernel/Regularizer/Square/ReadVariableOp2D
 encoder_2/BiasAdd/ReadVariableOp encoder_2/BiasAdd/ReadVariableOp2B
encoder_2/MatMul/ReadVariableOpencoder_2/MatMul/ReadVariableOp2h
2encoder_2/kernel/Regularizer/Square/ReadVariableOp2encoder_2/kernel/Regularizer/Square/ReadVariableOp2D
 encoder_3/BiasAdd/ReadVariableOp encoder_3/BiasAdd/ReadVariableOp2B
encoder_3/MatMul/ReadVariableOpencoder_3/MatMul/ReadVariableOp2h
2encoder_3/kernel/Regularizer/Square/ReadVariableOp2encoder_3/kernel/Regularizer/Square/ReadVariableOp28
reg_norm_1/AssignMovingAvgreg_norm_1/AssignMovingAvg2V
)reg_norm_1/AssignMovingAvg/ReadVariableOp)reg_norm_1/AssignMovingAvg/ReadVariableOp2<
reg_norm_1/AssignMovingAvg_1reg_norm_1/AssignMovingAvg_12Z
+reg_norm_1/AssignMovingAvg_1/ReadVariableOp+reg_norm_1/AssignMovingAvg_1/ReadVariableOp2J
#reg_norm_1/batchnorm/ReadVariableOp#reg_norm_1/batchnorm/ReadVariableOp2R
'reg_norm_1/batchnorm/mul/ReadVariableOp'reg_norm_1/batchnorm/mul/ReadVariableOp2D
 regress_1/BiasAdd/ReadVariableOp regress_1/BiasAdd/ReadVariableOp2B
regress_1/MatMul/ReadVariableOpregress_1/MatMul/ReadVariableOp2h
2regress_1/kernel/Regularizer/Square/ReadVariableOp2regress_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_regress_1_layer_call_and_return_conditional_losses_1358544

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2regress_1/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2�
2regress_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
#regress_1/kernel/Regularizer/SquareSquare:regress_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2s
"regress_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 regress_1/kernel/Regularizer/SumSum'regress_1/kernel/Regularizer/Square:y:0+regress_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"regress_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 regress_1/kernel/Regularizer/mulMul+regress_1/kernel/Regularizer/mul/x:output:0)regress_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^regress_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2regress_1/kernel/Regularizer/Square/ReadVariableOp2regress_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_elnorm_3_layer_call_and_return_conditional_losses_1358496

inputs+
mul_2_readvariableop_resource:P)
add_readvariableop_resource:P
identity��add/ReadVariableOp�mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������PJ
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:���������Pn
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:P*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:P*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PV
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������Pr
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
`
D__inference_dropout_layer_call_and_return_conditional_losses_1361316

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
/__inference_ebatch_norm_1_layer_call_fn_1360730

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1357875o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�
D__inference_model_1_layer_call_and_return_conditional_losses_1358613

inputs#
encoder_0_1358203:P
encoder_0_1358205:P#
ebatch_norm_0_1358208:P#
ebatch_norm_0_1358210:P#
ebatch_norm_0_1358212:P#
ebatch_norm_0_1358214:P
elnorm_0_1358260:P
elnorm_0_1358262:P#
encoder_1_1358282:PP
encoder_1_1358284:P#
ebatch_norm_1_1358287:P#
ebatch_norm_1_1358289:P#
ebatch_norm_1_1358291:P#
ebatch_norm_1_1358293:P
elnorm_1_1358339:P
elnorm_1_1358341:P#
encoder_2_1358361:PP
encoder_2_1358363:P#
ebatch_norm_2_1358366:P#
ebatch_norm_2_1358368:P#
ebatch_norm_2_1358370:P#
ebatch_norm_2_1358372:P
elnorm_2_1358418:P
elnorm_2_1358420:P#
encoder_3_1358440:PP
encoder_3_1358442:P#
ebatch_norm_3_1358445:P#
ebatch_norm_3_1358447:P#
ebatch_norm_3_1358449:P#
ebatch_norm_3_1358451:P
elnorm_3_1358497:P
elnorm_3_1358499:P%
ebottleneck_1358514:P!
ebottleneck_1358516:#
regress_1_1358545:2
regress_1_1358547:2 
reg_norm_1_1358550:2 
reg_norm_1_1358552:2 
reg_norm_1_1358554:2 
reg_norm_1_1358556:2
dense_1358577:2
dense_1358579:
identity��dense/StatefulPartitionedCall�%ebatch_norm_0/StatefulPartitionedCall�%ebatch_norm_1/StatefulPartitionedCall�%ebatch_norm_2/StatefulPartitionedCall�%ebatch_norm_3/StatefulPartitionedCall�#ebottleneck/StatefulPartitionedCall� elnorm_0/StatefulPartitionedCall� elnorm_1/StatefulPartitionedCall� elnorm_2/StatefulPartitionedCall� elnorm_3/StatefulPartitionedCall�!encoder_0/StatefulPartitionedCall�2encoder_0/kernel/Regularizer/Square/ReadVariableOp�!encoder_1/StatefulPartitionedCall�2encoder_1/kernel/Regularizer/Square/ReadVariableOp�!encoder_2/StatefulPartitionedCall�2encoder_2/kernel/Regularizer/Square/ReadVariableOp�!encoder_3/StatefulPartitionedCall�2encoder_3/kernel/Regularizer/Square/ReadVariableOp�"reg_norm_1/StatefulPartitionedCall�!regress_1/StatefulPartitionedCall�2regress_1/kernel/Regularizer/Square/ReadVariableOp�
!encoder_0/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_0_1358203encoder_0_1358205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_0_layer_call_and_return_conditional_losses_1358202�
%ebatch_norm_0/StatefulPartitionedCallStatefulPartitionedCall*encoder_0/StatefulPartitionedCall:output:0ebatch_norm_0_1358208ebatch_norm_0_1358210ebatch_norm_0_1358212ebatch_norm_0_1358214*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1357793�
 elnorm_0/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_0/StatefulPartitionedCall:output:0elnorm_0_1358260elnorm_0_1358262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_0_layer_call_and_return_conditional_losses_1358259�
!encoder_1/StatefulPartitionedCallStatefulPartitionedCall)elnorm_0/StatefulPartitionedCall:output:0encoder_1_1358282encoder_1_1358284*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_1_layer_call_and_return_conditional_losses_1358281�
%ebatch_norm_1/StatefulPartitionedCallStatefulPartitionedCall*encoder_1/StatefulPartitionedCall:output:0ebatch_norm_1_1358287ebatch_norm_1_1358289ebatch_norm_1_1358291ebatch_norm_1_1358293*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1357875�
 elnorm_1/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_1/StatefulPartitionedCall:output:0elnorm_1_1358339elnorm_1_1358341*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_1_layer_call_and_return_conditional_losses_1358338�
!encoder_2/StatefulPartitionedCallStatefulPartitionedCall)elnorm_1/StatefulPartitionedCall:output:0encoder_2_1358361encoder_2_1358363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_2_layer_call_and_return_conditional_losses_1358360�
%ebatch_norm_2/StatefulPartitionedCallStatefulPartitionedCall*encoder_2/StatefulPartitionedCall:output:0ebatch_norm_2_1358366ebatch_norm_2_1358368ebatch_norm_2_1358370ebatch_norm_2_1358372*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1357957�
 elnorm_2/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_2/StatefulPartitionedCall:output:0elnorm_2_1358418elnorm_2_1358420*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_2_layer_call_and_return_conditional_losses_1358417�
!encoder_3/StatefulPartitionedCallStatefulPartitionedCall)elnorm_2/StatefulPartitionedCall:output:0encoder_3_1358440encoder_3_1358442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_3_layer_call_and_return_conditional_losses_1358439�
%ebatch_norm_3/StatefulPartitionedCallStatefulPartitionedCall*encoder_3/StatefulPartitionedCall:output:0ebatch_norm_3_1358445ebatch_norm_3_1358447ebatch_norm_3_1358449ebatch_norm_3_1358451*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1358039�
 elnorm_3/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_3/StatefulPartitionedCall:output:0elnorm_3_1358497elnorm_3_1358499*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_3_layer_call_and_return_conditional_losses_1358496�
#ebottleneck/StatefulPartitionedCallStatefulPartitionedCall)elnorm_3/StatefulPartitionedCall:output:0ebottleneck_1358514ebottleneck_1358516*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_ebottleneck_layer_call_and_return_conditional_losses_1358513�
flatten/PartitionedCallPartitionedCall,ebottleneck/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1358525�
!regress_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0regress_1_1358545regress_1_1358547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_regress_1_layer_call_and_return_conditional_losses_1358544�
"reg_norm_1/StatefulPartitionedCallStatefulPartitionedCall*regress_1/StatefulPartitionedCall:output:0reg_norm_1_1358550reg_norm_1_1358552reg_norm_1_1358554reg_norm_1_1358556*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1358121�
dropout/PartitionedCallPartitionedCall+reg_norm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1358564�
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1358577dense_1358579*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1358576�
2encoder_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_0_1358203*
_output_shapes

:P*
dtype0�
#encoder_0/kernel/Regularizer/SquareSquare:encoder_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Ps
"encoder_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_0/kernel/Regularizer/SumSum'encoder_0/kernel/Regularizer/Square:y:0+encoder_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_0/kernel/Regularizer/mulMul+encoder_0/kernel/Regularizer/mul/x:output:0)encoder_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_1_1358282*
_output_shapes

:PP*
dtype0�
#encoder_1/kernel/Regularizer/SquareSquare:encoder_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_1/kernel/Regularizer/SumSum'encoder_1/kernel/Regularizer/Square:y:0+encoder_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_1/kernel/Regularizer/mulMul+encoder_1/kernel/Regularizer/mul/x:output:0)encoder_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_2_1358361*
_output_shapes

:PP*
dtype0�
#encoder_2/kernel/Regularizer/SquareSquare:encoder_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_2/kernel/Regularizer/SumSum'encoder_2/kernel/Regularizer/Square:y:0+encoder_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_2/kernel/Regularizer/mulMul+encoder_2/kernel/Regularizer/mul/x:output:0)encoder_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_3_1358440*
_output_shapes

:PP*
dtype0�
#encoder_3/kernel/Regularizer/SquareSquare:encoder_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_3/kernel/Regularizer/SumSum'encoder_3/kernel/Regularizer/Square:y:0+encoder_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_3/kernel/Regularizer/mulMul+encoder_3/kernel/Regularizer/mul/x:output:0)encoder_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2regress_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpregress_1_1358545*
_output_shapes

:2*
dtype0�
#regress_1/kernel/Regularizer/SquareSquare:regress_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2s
"regress_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 regress_1/kernel/Regularizer/SumSum'regress_1/kernel/Regularizer/Square:y:0+regress_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"regress_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 regress_1/kernel/Regularizer/mulMul+regress_1/kernel/Regularizer/mul/x:output:0)regress_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall&^ebatch_norm_0/StatefulPartitionedCall&^ebatch_norm_1/StatefulPartitionedCall&^ebatch_norm_2/StatefulPartitionedCall&^ebatch_norm_3/StatefulPartitionedCall$^ebottleneck/StatefulPartitionedCall!^elnorm_0/StatefulPartitionedCall!^elnorm_1/StatefulPartitionedCall!^elnorm_2/StatefulPartitionedCall!^elnorm_3/StatefulPartitionedCall"^encoder_0/StatefulPartitionedCall3^encoder_0/kernel/Regularizer/Square/ReadVariableOp"^encoder_1/StatefulPartitionedCall3^encoder_1/kernel/Regularizer/Square/ReadVariableOp"^encoder_2/StatefulPartitionedCall3^encoder_2/kernel/Regularizer/Square/ReadVariableOp"^encoder_3/StatefulPartitionedCall3^encoder_3/kernel/Regularizer/Square/ReadVariableOp#^reg_norm_1/StatefulPartitionedCall"^regress_1/StatefulPartitionedCall3^regress_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%ebatch_norm_0/StatefulPartitionedCall%ebatch_norm_0/StatefulPartitionedCall2N
%ebatch_norm_1/StatefulPartitionedCall%ebatch_norm_1/StatefulPartitionedCall2N
%ebatch_norm_2/StatefulPartitionedCall%ebatch_norm_2/StatefulPartitionedCall2N
%ebatch_norm_3/StatefulPartitionedCall%ebatch_norm_3/StatefulPartitionedCall2J
#ebottleneck/StatefulPartitionedCall#ebottleneck/StatefulPartitionedCall2D
 elnorm_0/StatefulPartitionedCall elnorm_0/StatefulPartitionedCall2D
 elnorm_1/StatefulPartitionedCall elnorm_1/StatefulPartitionedCall2D
 elnorm_2/StatefulPartitionedCall elnorm_2/StatefulPartitionedCall2D
 elnorm_3/StatefulPartitionedCall elnorm_3/StatefulPartitionedCall2F
!encoder_0/StatefulPartitionedCall!encoder_0/StatefulPartitionedCall2h
2encoder_0/kernel/Regularizer/Square/ReadVariableOp2encoder_0/kernel/Regularizer/Square/ReadVariableOp2F
!encoder_1/StatefulPartitionedCall!encoder_1/StatefulPartitionedCall2h
2encoder_1/kernel/Regularizer/Square/ReadVariableOp2encoder_1/kernel/Regularizer/Square/ReadVariableOp2F
!encoder_2/StatefulPartitionedCall!encoder_2/StatefulPartitionedCall2h
2encoder_2/kernel/Regularizer/Square/ReadVariableOp2encoder_2/kernel/Regularizer/Square/ReadVariableOp2F
!encoder_3/StatefulPartitionedCall!encoder_3/StatefulPartitionedCall2h
2encoder_3/kernel/Regularizer/Square/ReadVariableOp2encoder_3/kernel/Regularizer/Square/ReadVariableOp2H
"reg_norm_1/StatefulPartitionedCall"reg_norm_1/StatefulPartitionedCall2F
!regress_1/StatefulPartitionedCall!regress_1/StatefulPartitionedCall2h
2regress_1/kernel/Regularizer/Square/ReadVariableOp2regress_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1358004

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Ph
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�	
)__inference_model_1_layer_call_fn_1359841

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
	unknown_3:P
	unknown_4:P
	unknown_5:P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P

unknown_10:P

unknown_11:P

unknown_12:P

unknown_13:P

unknown_14:P

unknown_15:PP

unknown_16:P

unknown_17:P

unknown_18:P

unknown_19:P

unknown_20:P

unknown_21:P

unknown_22:P

unknown_23:PP

unknown_24:P

unknown_25:P

unknown_26:P

unknown_27:P

unknown_28:P

unknown_29:P

unknown_30:P

unknown_31:P

unknown_32:

unknown_33:2

unknown_34:2

unknown_35:2

unknown_36:2

unknown_37:2

unknown_38:2

unknown_39:2

unknown_40:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 	
 !"#$'()**6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1359058o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
H__inference_ebottleneck_layer_call_and_return_conditional_losses_1361180

inputs0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
F__inference_encoder_0_layer_call_and_return_conditional_losses_1358202

inputs0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2encoder_0/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
2encoder_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0�
#encoder_0/kernel/Regularizer/SquareSquare:encoder_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Ps
"encoder_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_0/kernel/Regularizer/SumSum'encoder_0/kernel/Regularizer/Square:y:0+encoder_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_0/kernel/Regularizer/mulMul+encoder_0/kernel/Regularizer/mul/x:output:0)encoder_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^encoder_0/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2encoder_0/kernel/Regularizer/Square/ReadVariableOp2encoder_0/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_elnorm_1_layer_call_and_return_conditional_losses_1360848

inputs+
mul_2_readvariableop_resource:P)
add_readvariableop_resource:P
identity��add/ReadVariableOp�mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������PJ
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:���������Pn
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:P*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:P*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PV
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������Pr
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
/__inference_ebatch_norm_3_layer_call_fn_1361055

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1358086o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
`
D__inference_dropout_layer_call_and_return_conditional_losses_1358722

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
E__inference_elnorm_3_layer_call_and_return_conditional_losses_1361160

inputs+
mul_2_readvariableop_resource:P)
add_readvariableop_resource:P
identity��add/ReadVariableOp�mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������PJ
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:���������Pn
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:P*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:P*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PV
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������Pr
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
,__inference_reg_norm_1_layer_call_fn_1361230

inputs
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1358121o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
*__inference_elnorm_2_layer_call_fn_1360962

inputs
unknown:P
	unknown_0:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_2_layer_call_and_return_conditional_losses_1358417o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�%
�
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1360797

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Ph
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
F__inference_encoder_2_layer_call_and_return_conditional_losses_1358360

inputs0
matmul_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2encoder_2/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
2encoder_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_2/kernel/Regularizer/SquareSquare:encoder_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_2/kernel/Regularizer/SumSum'encoder_2/kernel/Regularizer/Square:y:0+encoder_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_2/kernel/Regularizer/mulMul+encoder_2/kernel/Regularizer/mul/x:output:0)encoder_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^encoder_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2encoder_2/kernel/Regularizer/Square/ReadVariableOp2encoder_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�
D__inference_model_1_layer_call_and_return_conditional_losses_1359506
	intensity#
encoder_0_1359373:P
encoder_0_1359375:P#
ebatch_norm_0_1359378:P#
ebatch_norm_0_1359380:P#
ebatch_norm_0_1359382:P#
ebatch_norm_0_1359384:P
elnorm_0_1359387:P
elnorm_0_1359389:P#
encoder_1_1359392:PP
encoder_1_1359394:P#
ebatch_norm_1_1359397:P#
ebatch_norm_1_1359399:P#
ebatch_norm_1_1359401:P#
ebatch_norm_1_1359403:P
elnorm_1_1359406:P
elnorm_1_1359408:P#
encoder_2_1359411:PP
encoder_2_1359413:P#
ebatch_norm_2_1359416:P#
ebatch_norm_2_1359418:P#
ebatch_norm_2_1359420:P#
ebatch_norm_2_1359422:P
elnorm_2_1359425:P
elnorm_2_1359427:P#
encoder_3_1359430:PP
encoder_3_1359432:P#
ebatch_norm_3_1359435:P#
ebatch_norm_3_1359437:P#
ebatch_norm_3_1359439:P#
ebatch_norm_3_1359441:P
elnorm_3_1359444:P
elnorm_3_1359446:P%
ebottleneck_1359449:P!
ebottleneck_1359451:#
regress_1_1359455:2
regress_1_1359457:2 
reg_norm_1_1359460:2 
reg_norm_1_1359462:2 
reg_norm_1_1359464:2 
reg_norm_1_1359466:2
dense_1359470:2
dense_1359472:
identity��dense/StatefulPartitionedCall�%ebatch_norm_0/StatefulPartitionedCall�%ebatch_norm_1/StatefulPartitionedCall�%ebatch_norm_2/StatefulPartitionedCall�%ebatch_norm_3/StatefulPartitionedCall�#ebottleneck/StatefulPartitionedCall� elnorm_0/StatefulPartitionedCall� elnorm_1/StatefulPartitionedCall� elnorm_2/StatefulPartitionedCall� elnorm_3/StatefulPartitionedCall�!encoder_0/StatefulPartitionedCall�2encoder_0/kernel/Regularizer/Square/ReadVariableOp�!encoder_1/StatefulPartitionedCall�2encoder_1/kernel/Regularizer/Square/ReadVariableOp�!encoder_2/StatefulPartitionedCall�2encoder_2/kernel/Regularizer/Square/ReadVariableOp�!encoder_3/StatefulPartitionedCall�2encoder_3/kernel/Regularizer/Square/ReadVariableOp�"reg_norm_1/StatefulPartitionedCall�!regress_1/StatefulPartitionedCall�2regress_1/kernel/Regularizer/Square/ReadVariableOp�
!encoder_0/StatefulPartitionedCallStatefulPartitionedCall	intensityencoder_0_1359373encoder_0_1359375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_0_layer_call_and_return_conditional_losses_1358202�
%ebatch_norm_0/StatefulPartitionedCallStatefulPartitionedCall*encoder_0/StatefulPartitionedCall:output:0ebatch_norm_0_1359378ebatch_norm_0_1359380ebatch_norm_0_1359382ebatch_norm_0_1359384*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1357840�
 elnorm_0/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_0/StatefulPartitionedCall:output:0elnorm_0_1359387elnorm_0_1359389*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_0_layer_call_and_return_conditional_losses_1358259�
!encoder_1/StatefulPartitionedCallStatefulPartitionedCall)elnorm_0/StatefulPartitionedCall:output:0encoder_1_1359392encoder_1_1359394*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_1_layer_call_and_return_conditional_losses_1358281�
%ebatch_norm_1/StatefulPartitionedCallStatefulPartitionedCall*encoder_1/StatefulPartitionedCall:output:0ebatch_norm_1_1359397ebatch_norm_1_1359399ebatch_norm_1_1359401ebatch_norm_1_1359403*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1357922�
 elnorm_1/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_1/StatefulPartitionedCall:output:0elnorm_1_1359406elnorm_1_1359408*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_1_layer_call_and_return_conditional_losses_1358338�
!encoder_2/StatefulPartitionedCallStatefulPartitionedCall)elnorm_1/StatefulPartitionedCall:output:0encoder_2_1359411encoder_2_1359413*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_2_layer_call_and_return_conditional_losses_1358360�
%ebatch_norm_2/StatefulPartitionedCallStatefulPartitionedCall*encoder_2/StatefulPartitionedCall:output:0ebatch_norm_2_1359416ebatch_norm_2_1359418ebatch_norm_2_1359420ebatch_norm_2_1359422*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1358004�
 elnorm_2/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_2/StatefulPartitionedCall:output:0elnorm_2_1359425elnorm_2_1359427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_2_layer_call_and_return_conditional_losses_1358417�
!encoder_3/StatefulPartitionedCallStatefulPartitionedCall)elnorm_2/StatefulPartitionedCall:output:0encoder_3_1359430encoder_3_1359432*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_encoder_3_layer_call_and_return_conditional_losses_1358439�
%ebatch_norm_3/StatefulPartitionedCallStatefulPartitionedCall*encoder_3/StatefulPartitionedCall:output:0ebatch_norm_3_1359435ebatch_norm_3_1359437ebatch_norm_3_1359439ebatch_norm_3_1359441*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1358086�
 elnorm_3/StatefulPartitionedCallStatefulPartitionedCall.ebatch_norm_3/StatefulPartitionedCall:output:0elnorm_3_1359444elnorm_3_1359446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_3_layer_call_and_return_conditional_losses_1358496�
#ebottleneck/StatefulPartitionedCallStatefulPartitionedCall)elnorm_3/StatefulPartitionedCall:output:0ebottleneck_1359449ebottleneck_1359451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_ebottleneck_layer_call_and_return_conditional_losses_1358513�
flatten/PartitionedCallPartitionedCall,ebottleneck/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1358525�
!regress_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0regress_1_1359455regress_1_1359457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_regress_1_layer_call_and_return_conditional_losses_1358544�
"reg_norm_1/StatefulPartitionedCallStatefulPartitionedCall*regress_1/StatefulPartitionedCall:output:0reg_norm_1_1359460reg_norm_1_1359462reg_norm_1_1359464reg_norm_1_1359466*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1358168�
dropout/PartitionedCallPartitionedCall+reg_norm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1358722�
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1359470dense_1359472*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1358576�
2encoder_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_0_1359373*
_output_shapes

:P*
dtype0�
#encoder_0/kernel/Regularizer/SquareSquare:encoder_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Ps
"encoder_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_0/kernel/Regularizer/SumSum'encoder_0/kernel/Regularizer/Square:y:0+encoder_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_0/kernel/Regularizer/mulMul+encoder_0/kernel/Regularizer/mul/x:output:0)encoder_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_1_1359392*
_output_shapes

:PP*
dtype0�
#encoder_1/kernel/Regularizer/SquareSquare:encoder_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_1/kernel/Regularizer/SumSum'encoder_1/kernel/Regularizer/Square:y:0+encoder_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_1/kernel/Regularizer/mulMul+encoder_1/kernel/Regularizer/mul/x:output:0)encoder_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_2_1359411*
_output_shapes

:PP*
dtype0�
#encoder_2/kernel/Regularizer/SquareSquare:encoder_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_2/kernel/Regularizer/SumSum'encoder_2/kernel/Regularizer/Square:y:0+encoder_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_2/kernel/Regularizer/mulMul+encoder_2/kernel/Regularizer/mul/x:output:0)encoder_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2encoder_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpencoder_3_1359430*
_output_shapes

:PP*
dtype0�
#encoder_3/kernel/Regularizer/SquareSquare:encoder_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_3/kernel/Regularizer/SumSum'encoder_3/kernel/Regularizer/Square:y:0+encoder_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_3/kernel/Regularizer/mulMul+encoder_3/kernel/Regularizer/mul/x:output:0)encoder_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
2regress_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpregress_1_1359455*
_output_shapes

:2*
dtype0�
#regress_1/kernel/Regularizer/SquareSquare:regress_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2s
"regress_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 regress_1/kernel/Regularizer/SumSum'regress_1/kernel/Regularizer/Square:y:0+regress_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"regress_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 regress_1/kernel/Regularizer/mulMul+regress_1/kernel/Regularizer/mul/x:output:0)regress_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall&^ebatch_norm_0/StatefulPartitionedCall&^ebatch_norm_1/StatefulPartitionedCall&^ebatch_norm_2/StatefulPartitionedCall&^ebatch_norm_3/StatefulPartitionedCall$^ebottleneck/StatefulPartitionedCall!^elnorm_0/StatefulPartitionedCall!^elnorm_1/StatefulPartitionedCall!^elnorm_2/StatefulPartitionedCall!^elnorm_3/StatefulPartitionedCall"^encoder_0/StatefulPartitionedCall3^encoder_0/kernel/Regularizer/Square/ReadVariableOp"^encoder_1/StatefulPartitionedCall3^encoder_1/kernel/Regularizer/Square/ReadVariableOp"^encoder_2/StatefulPartitionedCall3^encoder_2/kernel/Regularizer/Square/ReadVariableOp"^encoder_3/StatefulPartitionedCall3^encoder_3/kernel/Regularizer/Square/ReadVariableOp#^reg_norm_1/StatefulPartitionedCall"^regress_1/StatefulPartitionedCall3^regress_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%ebatch_norm_0/StatefulPartitionedCall%ebatch_norm_0/StatefulPartitionedCall2N
%ebatch_norm_1/StatefulPartitionedCall%ebatch_norm_1/StatefulPartitionedCall2N
%ebatch_norm_2/StatefulPartitionedCall%ebatch_norm_2/StatefulPartitionedCall2N
%ebatch_norm_3/StatefulPartitionedCall%ebatch_norm_3/StatefulPartitionedCall2J
#ebottleneck/StatefulPartitionedCall#ebottleneck/StatefulPartitionedCall2D
 elnorm_0/StatefulPartitionedCall elnorm_0/StatefulPartitionedCall2D
 elnorm_1/StatefulPartitionedCall elnorm_1/StatefulPartitionedCall2D
 elnorm_2/StatefulPartitionedCall elnorm_2/StatefulPartitionedCall2D
 elnorm_3/StatefulPartitionedCall elnorm_3/StatefulPartitionedCall2F
!encoder_0/StatefulPartitionedCall!encoder_0/StatefulPartitionedCall2h
2encoder_0/kernel/Regularizer/Square/ReadVariableOp2encoder_0/kernel/Regularizer/Square/ReadVariableOp2F
!encoder_1/StatefulPartitionedCall!encoder_1/StatefulPartitionedCall2h
2encoder_1/kernel/Regularizer/Square/ReadVariableOp2encoder_1/kernel/Regularizer/Square/ReadVariableOp2F
!encoder_2/StatefulPartitionedCall!encoder_2/StatefulPartitionedCall2h
2encoder_2/kernel/Regularizer/Square/ReadVariableOp2encoder_2/kernel/Regularizer/Square/ReadVariableOp2F
!encoder_3/StatefulPartitionedCall!encoder_3/StatefulPartitionedCall2h
2encoder_3/kernel/Regularizer/Square/ReadVariableOp2encoder_3/kernel/Regularizer/Square/ReadVariableOp2H
"reg_norm_1/StatefulPartitionedCall"reg_norm_1/StatefulPartitionedCall2F
!regress_1/StatefulPartitionedCall!regress_1/StatefulPartitionedCall2h
2regress_1/kernel/Regularizer/Square/ReadVariableOp2regress_1/kernel/Regularizer/Square/ReadVariableOp:R N
'
_output_shapes
:���������
#
_user_specified_name	intensity
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_1361191

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_encoder_1_layer_call_and_return_conditional_losses_1360717

inputs0
matmul_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2encoder_1/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
2encoder_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_1/kernel/Regularizer/SquareSquare:encoder_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_1/kernel/Regularizer/SumSum'encoder_1/kernel/Regularizer/Square:y:0+encoder_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_1/kernel/Regularizer/mulMul+encoder_1/kernel/Regularizer/mul/x:output:0)encoder_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^encoder_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2encoder_1/kernel/Regularizer/Square/ReadVariableOp2encoder_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�%
�
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1361109

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Ph
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1361263

inputs/
!batchnorm_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:21
#batchnorm_readvariableop_1_resource:21
#batchnorm_readvariableop_2_resource:2
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
,__inference_reg_norm_1_layer_call_fn_1361243

inputs
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1358168o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
E__inference_elnorm_1_layer_call_and_return_conditional_losses_1358338

inputs+
mul_2_readvariableop_resource:P)
add_readvariableop_resource:P
identity��add/ReadVariableOp�mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������PJ
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:���������Pn
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:P*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:P*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PV
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������Pr
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
/__inference_ebatch_norm_0_layer_call_fn_1360574

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1357793o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
F__inference_encoder_2_layer_call_and_return_conditional_losses_1360873

inputs0
matmul_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2encoder_2/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
2encoder_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_2/kernel/Regularizer/SquareSquare:encoder_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_2/kernel/Regularizer/SumSum'encoder_2/kernel/Regularizer/Square:y:0+encoder_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_2/kernel/Regularizer/mulMul+encoder_2/kernel/Regularizer/mul/x:output:0)encoder_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^encoder_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2encoder_2/kernel/Regularizer/Square/ReadVariableOp2encoder_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�/
 __inference__traced_save_1361752
file_prefix/
+savev2_encoder_0_kernel_read_readvariableop-
)savev2_encoder_0_bias_read_readvariableop2
.savev2_ebatch_norm_0_gamma_read_readvariableop1
-savev2_ebatch_norm_0_beta_read_readvariableop8
4savev2_ebatch_norm_0_moving_mean_read_readvariableop<
8savev2_ebatch_norm_0_moving_variance_read_readvariableop-
)savev2_elnorm_0_gamma_read_readvariableop,
(savev2_elnorm_0_beta_read_readvariableop/
+savev2_encoder_1_kernel_read_readvariableop-
)savev2_encoder_1_bias_read_readvariableop2
.savev2_ebatch_norm_1_gamma_read_readvariableop1
-savev2_ebatch_norm_1_beta_read_readvariableop8
4savev2_ebatch_norm_1_moving_mean_read_readvariableop<
8savev2_ebatch_norm_1_moving_variance_read_readvariableop-
)savev2_elnorm_1_gamma_read_readvariableop,
(savev2_elnorm_1_beta_read_readvariableop/
+savev2_encoder_2_kernel_read_readvariableop-
)savev2_encoder_2_bias_read_readvariableop2
.savev2_ebatch_norm_2_gamma_read_readvariableop1
-savev2_ebatch_norm_2_beta_read_readvariableop8
4savev2_ebatch_norm_2_moving_mean_read_readvariableop<
8savev2_ebatch_norm_2_moving_variance_read_readvariableop-
)savev2_elnorm_2_gamma_read_readvariableop,
(savev2_elnorm_2_beta_read_readvariableop/
+savev2_encoder_3_kernel_read_readvariableop-
)savev2_encoder_3_bias_read_readvariableop2
.savev2_ebatch_norm_3_gamma_read_readvariableop1
-savev2_ebatch_norm_3_beta_read_readvariableop8
4savev2_ebatch_norm_3_moving_mean_read_readvariableop<
8savev2_ebatch_norm_3_moving_variance_read_readvariableop-
)savev2_elnorm_3_gamma_read_readvariableop,
(savev2_elnorm_3_beta_read_readvariableop1
-savev2_ebottleneck_kernel_read_readvariableop/
+savev2_ebottleneck_bias_read_readvariableop/
+savev2_regress_1_kernel_read_readvariableop-
)savev2_regress_1_bias_read_readvariableop/
+savev2_reg_norm_1_gamma_read_readvariableop.
*savev2_reg_norm_1_beta_read_readvariableop5
1savev2_reg_norm_1_moving_mean_read_readvariableop9
5savev2_reg_norm_1_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_encoder_0_kernel_m_read_readvariableop4
0savev2_adam_encoder_0_bias_m_read_readvariableop9
5savev2_adam_ebatch_norm_0_gamma_m_read_readvariableop8
4savev2_adam_ebatch_norm_0_beta_m_read_readvariableop4
0savev2_adam_elnorm_0_gamma_m_read_readvariableop3
/savev2_adam_elnorm_0_beta_m_read_readvariableop6
2savev2_adam_encoder_1_kernel_m_read_readvariableop4
0savev2_adam_encoder_1_bias_m_read_readvariableop9
5savev2_adam_ebatch_norm_1_gamma_m_read_readvariableop8
4savev2_adam_ebatch_norm_1_beta_m_read_readvariableop4
0savev2_adam_elnorm_1_gamma_m_read_readvariableop3
/savev2_adam_elnorm_1_beta_m_read_readvariableop6
2savev2_adam_encoder_2_kernel_m_read_readvariableop4
0savev2_adam_encoder_2_bias_m_read_readvariableop9
5savev2_adam_ebatch_norm_2_gamma_m_read_readvariableop8
4savev2_adam_ebatch_norm_2_beta_m_read_readvariableop4
0savev2_adam_elnorm_2_gamma_m_read_readvariableop3
/savev2_adam_elnorm_2_beta_m_read_readvariableop6
2savev2_adam_encoder_3_kernel_m_read_readvariableop4
0savev2_adam_encoder_3_bias_m_read_readvariableop9
5savev2_adam_ebatch_norm_3_gamma_m_read_readvariableop8
4savev2_adam_ebatch_norm_3_beta_m_read_readvariableop4
0savev2_adam_elnorm_3_gamma_m_read_readvariableop3
/savev2_adam_elnorm_3_beta_m_read_readvariableop8
4savev2_adam_ebottleneck_kernel_m_read_readvariableop6
2savev2_adam_ebottleneck_bias_m_read_readvariableop6
2savev2_adam_regress_1_kernel_m_read_readvariableop4
0savev2_adam_regress_1_bias_m_read_readvariableop6
2savev2_adam_reg_norm_1_gamma_m_read_readvariableop5
1savev2_adam_reg_norm_1_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop6
2savev2_adam_encoder_0_kernel_v_read_readvariableop4
0savev2_adam_encoder_0_bias_v_read_readvariableop9
5savev2_adam_ebatch_norm_0_gamma_v_read_readvariableop8
4savev2_adam_ebatch_norm_0_beta_v_read_readvariableop4
0savev2_adam_elnorm_0_gamma_v_read_readvariableop3
/savev2_adam_elnorm_0_beta_v_read_readvariableop6
2savev2_adam_encoder_1_kernel_v_read_readvariableop4
0savev2_adam_encoder_1_bias_v_read_readvariableop9
5savev2_adam_ebatch_norm_1_gamma_v_read_readvariableop8
4savev2_adam_ebatch_norm_1_beta_v_read_readvariableop4
0savev2_adam_elnorm_1_gamma_v_read_readvariableop3
/savev2_adam_elnorm_1_beta_v_read_readvariableop6
2savev2_adam_encoder_2_kernel_v_read_readvariableop4
0savev2_adam_encoder_2_bias_v_read_readvariableop9
5savev2_adam_ebatch_norm_2_gamma_v_read_readvariableop8
4savev2_adam_ebatch_norm_2_beta_v_read_readvariableop4
0savev2_adam_elnorm_2_gamma_v_read_readvariableop3
/savev2_adam_elnorm_2_beta_v_read_readvariableop6
2savev2_adam_encoder_3_kernel_v_read_readvariableop4
0savev2_adam_encoder_3_bias_v_read_readvariableop9
5savev2_adam_ebatch_norm_3_gamma_v_read_readvariableop8
4savev2_adam_ebatch_norm_3_beta_v_read_readvariableop4
0savev2_adam_elnorm_3_gamma_v_read_readvariableop3
/savev2_adam_elnorm_3_beta_v_read_readvariableop8
4savev2_adam_ebottleneck_kernel_v_read_readvariableop6
2savev2_adam_ebottleneck_bias_v_read_readvariableop6
2savev2_adam_regress_1_kernel_v_read_readvariableop4
0savev2_adam_regress_1_bias_v_read_readvariableop6
2savev2_adam_reg_norm_1_gamma_v_read_readvariableop5
1savev2_adam_reg_norm_1_beta_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
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
: �?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:r*
dtype0*�?
value�?B�?rB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:r*
dtype0*�
value�B�rB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �-
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_encoder_0_kernel_read_readvariableop)savev2_encoder_0_bias_read_readvariableop.savev2_ebatch_norm_0_gamma_read_readvariableop-savev2_ebatch_norm_0_beta_read_readvariableop4savev2_ebatch_norm_0_moving_mean_read_readvariableop8savev2_ebatch_norm_0_moving_variance_read_readvariableop)savev2_elnorm_0_gamma_read_readvariableop(savev2_elnorm_0_beta_read_readvariableop+savev2_encoder_1_kernel_read_readvariableop)savev2_encoder_1_bias_read_readvariableop.savev2_ebatch_norm_1_gamma_read_readvariableop-savev2_ebatch_norm_1_beta_read_readvariableop4savev2_ebatch_norm_1_moving_mean_read_readvariableop8savev2_ebatch_norm_1_moving_variance_read_readvariableop)savev2_elnorm_1_gamma_read_readvariableop(savev2_elnorm_1_beta_read_readvariableop+savev2_encoder_2_kernel_read_readvariableop)savev2_encoder_2_bias_read_readvariableop.savev2_ebatch_norm_2_gamma_read_readvariableop-savev2_ebatch_norm_2_beta_read_readvariableop4savev2_ebatch_norm_2_moving_mean_read_readvariableop8savev2_ebatch_norm_2_moving_variance_read_readvariableop)savev2_elnorm_2_gamma_read_readvariableop(savev2_elnorm_2_beta_read_readvariableop+savev2_encoder_3_kernel_read_readvariableop)savev2_encoder_3_bias_read_readvariableop.savev2_ebatch_norm_3_gamma_read_readvariableop-savev2_ebatch_norm_3_beta_read_readvariableop4savev2_ebatch_norm_3_moving_mean_read_readvariableop8savev2_ebatch_norm_3_moving_variance_read_readvariableop)savev2_elnorm_3_gamma_read_readvariableop(savev2_elnorm_3_beta_read_readvariableop-savev2_ebottleneck_kernel_read_readvariableop+savev2_ebottleneck_bias_read_readvariableop+savev2_regress_1_kernel_read_readvariableop)savev2_regress_1_bias_read_readvariableop+savev2_reg_norm_1_gamma_read_readvariableop*savev2_reg_norm_1_beta_read_readvariableop1savev2_reg_norm_1_moving_mean_read_readvariableop5savev2_reg_norm_1_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_encoder_0_kernel_m_read_readvariableop0savev2_adam_encoder_0_bias_m_read_readvariableop5savev2_adam_ebatch_norm_0_gamma_m_read_readvariableop4savev2_adam_ebatch_norm_0_beta_m_read_readvariableop0savev2_adam_elnorm_0_gamma_m_read_readvariableop/savev2_adam_elnorm_0_beta_m_read_readvariableop2savev2_adam_encoder_1_kernel_m_read_readvariableop0savev2_adam_encoder_1_bias_m_read_readvariableop5savev2_adam_ebatch_norm_1_gamma_m_read_readvariableop4savev2_adam_ebatch_norm_1_beta_m_read_readvariableop0savev2_adam_elnorm_1_gamma_m_read_readvariableop/savev2_adam_elnorm_1_beta_m_read_readvariableop2savev2_adam_encoder_2_kernel_m_read_readvariableop0savev2_adam_encoder_2_bias_m_read_readvariableop5savev2_adam_ebatch_norm_2_gamma_m_read_readvariableop4savev2_adam_ebatch_norm_2_beta_m_read_readvariableop0savev2_adam_elnorm_2_gamma_m_read_readvariableop/savev2_adam_elnorm_2_beta_m_read_readvariableop2savev2_adam_encoder_3_kernel_m_read_readvariableop0savev2_adam_encoder_3_bias_m_read_readvariableop5savev2_adam_ebatch_norm_3_gamma_m_read_readvariableop4savev2_adam_ebatch_norm_3_beta_m_read_readvariableop0savev2_adam_elnorm_3_gamma_m_read_readvariableop/savev2_adam_elnorm_3_beta_m_read_readvariableop4savev2_adam_ebottleneck_kernel_m_read_readvariableop2savev2_adam_ebottleneck_bias_m_read_readvariableop2savev2_adam_regress_1_kernel_m_read_readvariableop0savev2_adam_regress_1_bias_m_read_readvariableop2savev2_adam_reg_norm_1_gamma_m_read_readvariableop1savev2_adam_reg_norm_1_beta_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop2savev2_adam_encoder_0_kernel_v_read_readvariableop0savev2_adam_encoder_0_bias_v_read_readvariableop5savev2_adam_ebatch_norm_0_gamma_v_read_readvariableop4savev2_adam_ebatch_norm_0_beta_v_read_readvariableop0savev2_adam_elnorm_0_gamma_v_read_readvariableop/savev2_adam_elnorm_0_beta_v_read_readvariableop2savev2_adam_encoder_1_kernel_v_read_readvariableop0savev2_adam_encoder_1_bias_v_read_readvariableop5savev2_adam_ebatch_norm_1_gamma_v_read_readvariableop4savev2_adam_ebatch_norm_1_beta_v_read_readvariableop0savev2_adam_elnorm_1_gamma_v_read_readvariableop/savev2_adam_elnorm_1_beta_v_read_readvariableop2savev2_adam_encoder_2_kernel_v_read_readvariableop0savev2_adam_encoder_2_bias_v_read_readvariableop5savev2_adam_ebatch_norm_2_gamma_v_read_readvariableop4savev2_adam_ebatch_norm_2_beta_v_read_readvariableop0savev2_adam_elnorm_2_gamma_v_read_readvariableop/savev2_adam_elnorm_2_beta_v_read_readvariableop2savev2_adam_encoder_3_kernel_v_read_readvariableop0savev2_adam_encoder_3_bias_v_read_readvariableop5savev2_adam_ebatch_norm_3_gamma_v_read_readvariableop4savev2_adam_ebatch_norm_3_beta_v_read_readvariableop0savev2_adam_elnorm_3_gamma_v_read_readvariableop/savev2_adam_elnorm_3_beta_v_read_readvariableop4savev2_adam_ebottleneck_kernel_v_read_readvariableop2savev2_adam_ebottleneck_bias_v_read_readvariableop2savev2_adam_regress_1_kernel_v_read_readvariableop0savev2_adam_regress_1_bias_v_read_readvariableop2savev2_adam_reg_norm_1_gamma_v_read_readvariableop1savev2_adam_reg_norm_1_beta_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypesv
t2r	�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :P:P:P:P:P:P:P:P:PP:P:P:P:P:P:P:P:PP:P:P:P:P:P:P:P:PP:P:P:P:P:P:P:P:P::2:2:2:2:2:2:2:: : : : : : : :P:P:P:P:P:P:PP:P:P:P:P:P:PP:P:P:P:P:P:PP:P:P:P:P:P:P::2:2:2:2:2::P:P:P:P:P:P:PP:P:P:P:P:P:PP:P:P:P:P:P:PP:P:P:P:P:P:P::2:2:2:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P:$	 

_output_shapes

:PP: 


_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P:  

_output_shapes
:P:$! 

_output_shapes

:P: "

_output_shapes
::$# 

_output_shapes

:2: $

_output_shapes
:2: %

_output_shapes
:2: &

_output_shapes
:2: '

_output_shapes
:2: (

_output_shapes
:2:$) 

_output_shapes

:2: *

_output_shapes
::+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :$2 

_output_shapes

:P: 3

_output_shapes
:P: 4

_output_shapes
:P: 5

_output_shapes
:P: 6

_output_shapes
:P: 7

_output_shapes
:P:$8 

_output_shapes

:PP: 9

_output_shapes
:P: :

_output_shapes
:P: ;

_output_shapes
:P: <

_output_shapes
:P: =

_output_shapes
:P:$> 

_output_shapes

:PP: ?

_output_shapes
:P: @

_output_shapes
:P: A

_output_shapes
:P: B

_output_shapes
:P: C

_output_shapes
:P:$D 

_output_shapes

:PP: E

_output_shapes
:P: F

_output_shapes
:P: G

_output_shapes
:P: H

_output_shapes
:P: I

_output_shapes
:P:$J 

_output_shapes

:P: K

_output_shapes
::$L 

_output_shapes

:2: M

_output_shapes
:2: N

_output_shapes
:2: O

_output_shapes
:2:$P 

_output_shapes

:2: Q

_output_shapes
::$R 

_output_shapes

:P: S

_output_shapes
:P: T

_output_shapes
:P: U

_output_shapes
:P: V

_output_shapes
:P: W

_output_shapes
:P:$X 

_output_shapes

:PP: Y

_output_shapes
:P: Z

_output_shapes
:P: [

_output_shapes
:P: \

_output_shapes
:P: ]

_output_shapes
:P:$^ 

_output_shapes

:PP: _

_output_shapes
:P: `

_output_shapes
:P: a

_output_shapes
:P: b

_output_shapes
:P: c

_output_shapes
:P:$d 

_output_shapes

:PP: e

_output_shapes
:P: f

_output_shapes
:P: g

_output_shapes
:P: h

_output_shapes
:P: i

_output_shapes
:P:$j 

_output_shapes

:P: k

_output_shapes
::$l 

_output_shapes

:2: m

_output_shapes
:2: n

_output_shapes
:2: o

_output_shapes
:2:$p 

_output_shapes

:2: q

_output_shapes
::r

_output_shapes
: 
�
�
'__inference_dense_layer_call_fn_1361325

inputs
unknown:2
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1358576o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
*__inference_elnorm_0_layer_call_fn_1360650

inputs
unknown:P
	unknown_0:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_0_layer_call_and_return_conditional_losses_1358259o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_1358564

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�%
�
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1358168

inputs5
'assignmovingavg_readvariableop_resource:27
)assignmovingavg_1_readvariableop_resource:23
%batchnorm_mul_readvariableop_resource:2/
!batchnorm_readvariableop_resource:2
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:2*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:2*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
E__inference_elnorm_2_layer_call_and_return_conditional_losses_1358417

inputs+
mul_2_readvariableop_resource:P)
add_readvariableop_resource:P
identity��add/ReadVariableOp�mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������PJ
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:���������Pn
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:P*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:P*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PV
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������Pr
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
/__inference_ebatch_norm_2_layer_call_fn_1360886

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1357957o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
E
)__inference_dropout_layer_call_fn_1361302

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1358564`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_1361357M
;encoder_1_kernel_regularizer_square_readvariableop_resource:PP
identity��2encoder_1/kernel/Regularizer/Square/ReadVariableOp�
2encoder_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;encoder_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:PP*
dtype0�
#encoder_1/kernel/Regularizer/SquareSquare:encoder_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PPs
"encoder_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 encoder_1/kernel/Regularizer/SumSum'encoder_1/kernel/Regularizer/Square:y:0+encoder_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"encoder_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
 encoder_1/kernel/Regularizer/mulMul+encoder_1/kernel/Regularizer/mul/x:output:0)encoder_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$encoder_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^encoder_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2encoder_1/kernel/Regularizer/Square/ReadVariableOp2encoder_1/kernel/Regularizer/Square/ReadVariableOp
�
�
E__inference_elnorm_0_layer_call_and_return_conditional_losses_1358259

inputs+
mul_2_readvariableop_resource:P)
add_readvariableop_resource:P
identity��add/ReadVariableOp�mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������PJ
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:���������Pn
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:P*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:P*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PV
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������Pr
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�%
�
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1360641

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:P*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Ph
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1361075

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Pz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
*__inference_elnorm_3_layer_call_fn_1361118

inputs
unknown:P
	unknown_0:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_elnorm_3_layer_call_and_return_conditional_losses_1358496o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
E__inference_elnorm_2_layer_call_and_return_conditional_losses_1361004

inputs+
mul_2_readvariableop_resource:P)
add_readvariableop_resource:P
identity��add/ReadVariableOp�mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������PJ
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������P:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:���������Pn
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:P*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:P*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PV
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������Pr
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1357875

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Pz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_1358525

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
	intensity2
serving_default_intensity:0���������9
dense0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
layer_with_weights-11
layer-12
layer_with_weights-12
layer-13
layer-14
layer_with_weights-13
layer-15
layer_with_weights-14
layer-16
layer-17
layer_with_weights-15
layer-18
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+axis
	,gamma
-beta
.moving_mean
/moving_variance"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6axis
	7gamma
8beta"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias"
_tf_keras_layer
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance"
_tf_keras_layer
�
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Raxis
	Sgamma
Tbeta"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias"
_tf_keras_layer
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance"
_tf_keras_layer
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses
naxis
	ogamma
pbeta"
_tf_keras_layer
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias"
_tf_keras_layer
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
#0
$1
,2
-3
.4
/5
76
87
?8
@9
H10
I11
J12
K13
S14
T15
[16
\17
d18
e19
f20
g21
o22
p23
w24
x25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41"
trackable_list_wrapper
�
#0
$1
,2
-3
74
85
?6
@7
H8
I9
S10
T11
[12
\13
d14
e15
o16
p17
w18
x19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31"
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
)__inference_model_1_layer_call_fn_1358700
)__inference_model_1_layer_call_fn_1359752
)__inference_model_1_layer_call_fn_1359841
)__inference_model_1_layer_call_fn_1359234�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
D__inference_model_1_layer_call_and_return_conditional_losses_1360154
D__inference_model_1_layer_call_and_return_conditional_losses_1360536
D__inference_model_1_layer_call_and_return_conditional_losses_1359370
D__inference_model_1_layer_call_and_return_conditional_losses_1359506�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
"__inference__wrapped_model_1357769	intensity"�
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
�
�beta_1
�beta_2

�decay
�learning_rate
	�iter#m�$m�,m�-m�7m�8m�?m�@m�Hm�Im�Sm�Tm�[m�\m�dm�em�om�pm�wm�xm�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�#v�$v�,v�-v�7v�8v�?v�@v�Hv�Iv�Sv�Tv�[v�\v�dv�ev�ov�pv�wv�xv�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
-
�serving_default"
signature_map
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_encoder_0_layer_call_fn_1360545�
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
 z�trace_0
�
�trace_02�
F__inference_encoder_0_layer_call_and_return_conditional_losses_1360561�
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
 z�trace_0
": P2encoder_0/kernel
:P2encoder_0/bias
<
,0
-1
.2
/3"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
/__inference_ebatch_norm_0_layer_call_fn_1360574
/__inference_ebatch_norm_0_layer_call_fn_1360587�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1360607
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1360641�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
!:P2ebatch_norm_0/gamma
 :P2ebatch_norm_0/beta
):'P (2ebatch_norm_0/moving_mean
-:+P (2ebatch_norm_0/moving_variance
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_elnorm_0_layer_call_fn_1360650�
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
 z�trace_0
�
�trace_02�
E__inference_elnorm_0_layer_call_and_return_conditional_losses_1360692�
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
 z�trace_0
 "
trackable_list_wrapper
:P2elnorm_0/gamma
:P2elnorm_0/beta
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_encoder_1_layer_call_fn_1360701�
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
 z�trace_0
�
�trace_02�
F__inference_encoder_1_layer_call_and_return_conditional_losses_1360717�
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
 z�trace_0
": PP2encoder_1/kernel
:P2encoder_1/bias
<
H0
I1
J2
K3"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
/__inference_ebatch_norm_1_layer_call_fn_1360730
/__inference_ebatch_norm_1_layer_call_fn_1360743�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1360763
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1360797�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
!:P2ebatch_norm_1/gamma
 :P2ebatch_norm_1/beta
):'P (2ebatch_norm_1/moving_mean
-:+P (2ebatch_norm_1/moving_variance
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_elnorm_1_layer_call_fn_1360806�
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
 z�trace_0
�
�trace_02�
E__inference_elnorm_1_layer_call_and_return_conditional_losses_1360848�
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
 z�trace_0
 "
trackable_list_wrapper
:P2elnorm_1/gamma
:P2elnorm_1/beta
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_encoder_2_layer_call_fn_1360857�
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
 z�trace_0
�
�trace_02�
F__inference_encoder_2_layer_call_and_return_conditional_losses_1360873�
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
 z�trace_0
": PP2encoder_2/kernel
:P2encoder_2/bias
<
d0
e1
f2
g3"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
/__inference_ebatch_norm_2_layer_call_fn_1360886
/__inference_ebatch_norm_2_layer_call_fn_1360899�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1360919
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1360953�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
!:P2ebatch_norm_2/gamma
 :P2ebatch_norm_2/beta
):'P (2ebatch_norm_2/moving_mean
-:+P (2ebatch_norm_2/moving_variance
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_elnorm_2_layer_call_fn_1360962�
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
 z�trace_0
�
�trace_02�
E__inference_elnorm_2_layer_call_and_return_conditional_losses_1361004�
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
 z�trace_0
 "
trackable_list_wrapper
:P2elnorm_2/gamma
:P2elnorm_2/beta
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_encoder_3_layer_call_fn_1361013�
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
 z�trace_0
�
�trace_02�
F__inference_encoder_3_layer_call_and_return_conditional_losses_1361029�
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
 z�trace_0
": PP2encoder_3/kernel
:P2encoder_3/bias
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
/__inference_ebatch_norm_3_layer_call_fn_1361042
/__inference_ebatch_norm_3_layer_call_fn_1361055�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1361075
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1361109�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
!:P2ebatch_norm_3/gamma
 :P2ebatch_norm_3/beta
):'P (2ebatch_norm_3/moving_mean
-:+P (2ebatch_norm_3/moving_variance
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_elnorm_3_layer_call_fn_1361118�
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
 z�trace_0
�
�trace_02�
E__inference_elnorm_3_layer_call_and_return_conditional_losses_1361160�
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
 z�trace_0
 "
trackable_list_wrapper
:P2elnorm_3/gamma
:P2elnorm_3/beta
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_ebottleneck_layer_call_fn_1361169�
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
 z�trace_0
�
�trace_02�
H__inference_ebottleneck_layer_call_and_return_conditional_losses_1361180�
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
 z�trace_0
$:"P2ebottleneck/kernel
:2ebottleneck/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_flatten_layer_call_fn_1361185�
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
 z�trace_0
�
�trace_02�
D__inference_flatten_layer_call_and_return_conditional_losses_1361191�
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
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_regress_1_layer_call_fn_1361200�
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
 z�trace_0
�
�trace_02�
F__inference_regress_1_layer_call_and_return_conditional_losses_1361217�
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
 z�trace_0
": 22regress_1/kernel
:22regress_1/bias
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_reg_norm_1_layer_call_fn_1361230
,__inference_reg_norm_1_layer_call_fn_1361243�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1361263
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1361297�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
:22reg_norm_1/gamma
:22reg_norm_1/beta
&:$2 (2reg_norm_1/moving_mean
*:(2 (2reg_norm_1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_dropout_layer_call_fn_1361302
)__inference_dropout_layer_call_fn_1361307�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
D__inference_dropout_layer_call_and_return_conditional_losses_1361312
D__inference_dropout_layer_call_and_return_conditional_losses_1361316�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_layer_call_fn_1361325�
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
 z�trace_0
�
�trace_02�
B__inference_dense_layer_call_and_return_conditional_losses_1361335�
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
 z�trace_0
:22dense/kernel
:2
dense/bias
�
�trace_02�
__inference_loss_fn_0_1361346�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_1361357�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_1361368�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_1361379�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_1361390�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
j
.0
/1
J2
K3
f4
g5
�6
�7
�8
�9"
trackable_list_wrapper
�
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
11
12
13
14
15
16
17
18"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_model_1_layer_call_fn_1358700	intensity"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
)__inference_model_1_layer_call_fn_1359752inputs"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
)__inference_model_1_layer_call_fn_1359841inputs"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
)__inference_model_1_layer_call_fn_1359234	intensity"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
D__inference_model_1_layer_call_and_return_conditional_losses_1360154inputs"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
D__inference_model_1_layer_call_and_return_conditional_losses_1360536inputs"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
D__inference_model_1_layer_call_and_return_conditional_losses_1359370	intensity"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
D__inference_model_1_layer_call_and_return_conditional_losses_1359506	intensity"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
�B�
%__inference_signature_wrapper_1359633	intensity"�
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
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_encoder_0_layer_call_fn_1360545inputs"�
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
F__inference_encoder_0_layer_call_and_return_conditional_losses_1360561inputs"�
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
.
.0
/1"
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
/__inference_ebatch_norm_0_layer_call_fn_1360574inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
/__inference_ebatch_norm_0_layer_call_fn_1360587inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1360607inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1360641inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
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
*__inference_elnorm_0_layer_call_fn_1360650inputs"�
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
E__inference_elnorm_0_layer_call_and_return_conditional_losses_1360692inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_encoder_1_layer_call_fn_1360701inputs"�
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
F__inference_encoder_1_layer_call_and_return_conditional_losses_1360717inputs"�
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
.
J0
K1"
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
/__inference_ebatch_norm_1_layer_call_fn_1360730inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
/__inference_ebatch_norm_1_layer_call_fn_1360743inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1360763inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1360797inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
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
*__inference_elnorm_1_layer_call_fn_1360806inputs"�
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
E__inference_elnorm_1_layer_call_and_return_conditional_losses_1360848inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_encoder_2_layer_call_fn_1360857inputs"�
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
F__inference_encoder_2_layer_call_and_return_conditional_losses_1360873inputs"�
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
.
f0
g1"
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
/__inference_ebatch_norm_2_layer_call_fn_1360886inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
/__inference_ebatch_norm_2_layer_call_fn_1360899inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1360919inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1360953inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
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
*__inference_elnorm_2_layer_call_fn_1360962inputs"�
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
E__inference_elnorm_2_layer_call_and_return_conditional_losses_1361004inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_encoder_3_layer_call_fn_1361013inputs"�
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
F__inference_encoder_3_layer_call_and_return_conditional_losses_1361029inputs"�
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
0
�0
�1"
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
/__inference_ebatch_norm_3_layer_call_fn_1361042inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
/__inference_ebatch_norm_3_layer_call_fn_1361055inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1361075inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1361109inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
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
*__inference_elnorm_3_layer_call_fn_1361118inputs"�
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
E__inference_elnorm_3_layer_call_and_return_conditional_losses_1361160inputs"�
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
-__inference_ebottleneck_layer_call_fn_1361169inputs"�
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
H__inference_ebottleneck_layer_call_and_return_conditional_losses_1361180inputs"�
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
)__inference_flatten_layer_call_fn_1361185inputs"�
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
D__inference_flatten_layer_call_and_return_conditional_losses_1361191inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_regress_1_layer_call_fn_1361200inputs"�
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
F__inference_regress_1_layer_call_and_return_conditional_losses_1361217inputs"�
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
0
�0
�1"
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
,__inference_reg_norm_1_layer_call_fn_1361230inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
,__inference_reg_norm_1_layer_call_fn_1361243inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1361263inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1361297inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
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
)__inference_dropout_layer_call_fn_1361302inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
)__inference_dropout_layer_call_fn_1361307inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
D__inference_dropout_layer_call_and_return_conditional_losses_1361312inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
D__inference_dropout_layer_call_and_return_conditional_losses_1361316inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
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
'__inference_dense_layer_call_fn_1361325inputs"�
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
B__inference_dense_layer_call_and_return_conditional_losses_1361335inputs"�
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
__inference_loss_fn_0_1361346"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_1361357"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_1361368"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_1361379"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_4_1361390"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
':%P2Adam/encoder_0/kernel/m
!:P2Adam/encoder_0/bias/m
&:$P2Adam/ebatch_norm_0/gamma/m
%:#P2Adam/ebatch_norm_0/beta/m
!:P2Adam/elnorm_0/gamma/m
 :P2Adam/elnorm_0/beta/m
':%PP2Adam/encoder_1/kernel/m
!:P2Adam/encoder_1/bias/m
&:$P2Adam/ebatch_norm_1/gamma/m
%:#P2Adam/ebatch_norm_1/beta/m
!:P2Adam/elnorm_1/gamma/m
 :P2Adam/elnorm_1/beta/m
':%PP2Adam/encoder_2/kernel/m
!:P2Adam/encoder_2/bias/m
&:$P2Adam/ebatch_norm_2/gamma/m
%:#P2Adam/ebatch_norm_2/beta/m
!:P2Adam/elnorm_2/gamma/m
 :P2Adam/elnorm_2/beta/m
':%PP2Adam/encoder_3/kernel/m
!:P2Adam/encoder_3/bias/m
&:$P2Adam/ebatch_norm_3/gamma/m
%:#P2Adam/ebatch_norm_3/beta/m
!:P2Adam/elnorm_3/gamma/m
 :P2Adam/elnorm_3/beta/m
):'P2Adam/ebottleneck/kernel/m
#:!2Adam/ebottleneck/bias/m
':%22Adam/regress_1/kernel/m
!:22Adam/regress_1/bias/m
#:!22Adam/reg_norm_1/gamma/m
": 22Adam/reg_norm_1/beta/m
#:!22Adam/dense/kernel/m
:2Adam/dense/bias/m
':%P2Adam/encoder_0/kernel/v
!:P2Adam/encoder_0/bias/v
&:$P2Adam/ebatch_norm_0/gamma/v
%:#P2Adam/ebatch_norm_0/beta/v
!:P2Adam/elnorm_0/gamma/v
 :P2Adam/elnorm_0/beta/v
':%PP2Adam/encoder_1/kernel/v
!:P2Adam/encoder_1/bias/v
&:$P2Adam/ebatch_norm_1/gamma/v
%:#P2Adam/ebatch_norm_1/beta/v
!:P2Adam/elnorm_1/gamma/v
 :P2Adam/elnorm_1/beta/v
':%PP2Adam/encoder_2/kernel/v
!:P2Adam/encoder_2/bias/v
&:$P2Adam/ebatch_norm_2/gamma/v
%:#P2Adam/ebatch_norm_2/beta/v
!:P2Adam/elnorm_2/gamma/v
 :P2Adam/elnorm_2/beta/v
':%PP2Adam/encoder_3/kernel/v
!:P2Adam/encoder_3/bias/v
&:$P2Adam/ebatch_norm_3/gamma/v
%:#P2Adam/ebatch_norm_3/beta/v
!:P2Adam/elnorm_3/gamma/v
 :P2Adam/elnorm_3/beta/v
):'P2Adam/ebottleneck/kernel/v
#:!2Adam/ebottleneck/bias/v
':%22Adam/regress_1/kernel/v
!:22Adam/regress_1/bias/v
#:!22Adam/reg_norm_1/gamma/v
": 22Adam/reg_norm_1/beta/v
#:!22Adam/dense/kernel/v
:2Adam/dense/bias/v�
"__inference__wrapped_model_1357769�:#$/,.-78?@KHJIST[\gdfeopwx����������������2�/
(�%
#� 
	intensity���������
� "-�*
(
dense�
dense����������
B__inference_dense_layer_call_and_return_conditional_losses_1361335^��/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� |
'__inference_dense_layer_call_fn_1361325Q��/�,
%�"
 �
inputs���������2
� "�����������
D__inference_dropout_layer_call_and_return_conditional_losses_1361312\3�0
)�&
 �
inputs���������2
p 
� "%�"
�
0���������2
� �
D__inference_dropout_layer_call_and_return_conditional_losses_1361316\3�0
)�&
 �
inputs���������2
p
� "%�"
�
0���������2
� |
)__inference_dropout_layer_call_fn_1361302O3�0
)�&
 �
inputs���������2
p 
� "����������2|
)__inference_dropout_layer_call_fn_1361307O3�0
)�&
 �
inputs���������2
p
� "����������2�
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1360607b/,.-3�0
)�&
 �
inputs���������P
p 
� "%�"
�
0���������P
� �
J__inference_ebatch_norm_0_layer_call_and_return_conditional_losses_1360641b./,-3�0
)�&
 �
inputs���������P
p
� "%�"
�
0���������P
� �
/__inference_ebatch_norm_0_layer_call_fn_1360574U/,.-3�0
)�&
 �
inputs���������P
p 
� "����������P�
/__inference_ebatch_norm_0_layer_call_fn_1360587U./,-3�0
)�&
 �
inputs���������P
p
� "����������P�
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1360763bKHJI3�0
)�&
 �
inputs���������P
p 
� "%�"
�
0���������P
� �
J__inference_ebatch_norm_1_layer_call_and_return_conditional_losses_1360797bJKHI3�0
)�&
 �
inputs���������P
p
� "%�"
�
0���������P
� �
/__inference_ebatch_norm_1_layer_call_fn_1360730UKHJI3�0
)�&
 �
inputs���������P
p 
� "����������P�
/__inference_ebatch_norm_1_layer_call_fn_1360743UJKHI3�0
)�&
 �
inputs���������P
p
� "����������P�
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1360919bgdfe3�0
)�&
 �
inputs���������P
p 
� "%�"
�
0���������P
� �
J__inference_ebatch_norm_2_layer_call_and_return_conditional_losses_1360953bfgde3�0
)�&
 �
inputs���������P
p
� "%�"
�
0���������P
� �
/__inference_ebatch_norm_2_layer_call_fn_1360886Ugdfe3�0
)�&
 �
inputs���������P
p 
� "����������P�
/__inference_ebatch_norm_2_layer_call_fn_1360899Ufgde3�0
)�&
 �
inputs���������P
p
� "����������P�
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1361075f����3�0
)�&
 �
inputs���������P
p 
� "%�"
�
0���������P
� �
J__inference_ebatch_norm_3_layer_call_and_return_conditional_losses_1361109f����3�0
)�&
 �
inputs���������P
p
� "%�"
�
0���������P
� �
/__inference_ebatch_norm_3_layer_call_fn_1361042Y����3�0
)�&
 �
inputs���������P
p 
� "����������P�
/__inference_ebatch_norm_3_layer_call_fn_1361055Y����3�0
)�&
 �
inputs���������P
p
� "����������P�
H__inference_ebottleneck_layer_call_and_return_conditional_losses_1361180^��/�,
%�"
 �
inputs���������P
� "%�"
�
0���������
� �
-__inference_ebottleneck_layer_call_fn_1361169Q��/�,
%�"
 �
inputs���������P
� "�����������
E__inference_elnorm_0_layer_call_and_return_conditional_losses_1360692\78/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� }
*__inference_elnorm_0_layer_call_fn_1360650O78/�,
%�"
 �
inputs���������P
� "����������P�
E__inference_elnorm_1_layer_call_and_return_conditional_losses_1360848\ST/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� }
*__inference_elnorm_1_layer_call_fn_1360806OST/�,
%�"
 �
inputs���������P
� "����������P�
E__inference_elnorm_2_layer_call_and_return_conditional_losses_1361004\op/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� }
*__inference_elnorm_2_layer_call_fn_1360962Oop/�,
%�"
 �
inputs���������P
� "����������P�
E__inference_elnorm_3_layer_call_and_return_conditional_losses_1361160^��/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� 
*__inference_elnorm_3_layer_call_fn_1361118Q��/�,
%�"
 �
inputs���������P
� "����������P�
F__inference_encoder_0_layer_call_and_return_conditional_losses_1360561\#$/�,
%�"
 �
inputs���������
� "%�"
�
0���������P
� ~
+__inference_encoder_0_layer_call_fn_1360545O#$/�,
%�"
 �
inputs���������
� "����������P�
F__inference_encoder_1_layer_call_and_return_conditional_losses_1360717\?@/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� ~
+__inference_encoder_1_layer_call_fn_1360701O?@/�,
%�"
 �
inputs���������P
� "����������P�
F__inference_encoder_2_layer_call_and_return_conditional_losses_1360873\[\/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� ~
+__inference_encoder_2_layer_call_fn_1360857O[\/�,
%�"
 �
inputs���������P
� "����������P�
F__inference_encoder_3_layer_call_and_return_conditional_losses_1361029\wx/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� ~
+__inference_encoder_3_layer_call_fn_1361013Owx/�,
%�"
 �
inputs���������P
� "����������P�
D__inference_flatten_layer_call_and_return_conditional_losses_1361191X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� x
)__inference_flatten_layer_call_fn_1361185K/�,
%�"
 �
inputs���������
� "����������<
__inference_loss_fn_0_1361346#�

� 
� "� <
__inference_loss_fn_1_1361357?�

� 
� "� <
__inference_loss_fn_2_1361368[�

� 
� "� <
__inference_loss_fn_3_1361379w�

� 
� "� =
__inference_loss_fn_4_1361390��

� 
� "� �
D__inference_model_1_layer_call_and_return_conditional_losses_1359370�:#$/,.-78?@KHJIST[\gdfeopwx����������������:�7
0�-
#� 
	intensity���������
p 

 
� "%�"
�
0���������
� �
D__inference_model_1_layer_call_and_return_conditional_losses_1359506�:#$./,-78?@JKHIST[\fgdeopwx����������������:�7
0�-
#� 
	intensity���������
p

 
� "%�"
�
0���������
� �
D__inference_model_1_layer_call_and_return_conditional_losses_1360154�:#$/,.-78?@KHJIST[\gdfeopwx����������������7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
D__inference_model_1_layer_call_and_return_conditional_losses_1360536�:#$./,-78?@JKHIST[\fgdeopwx����������������7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
)__inference_model_1_layer_call_fn_1358700�:#$/,.-78?@KHJIST[\gdfeopwx����������������:�7
0�-
#� 
	intensity���������
p 

 
� "�����������
)__inference_model_1_layer_call_fn_1359234�:#$./,-78?@JKHIST[\fgdeopwx����������������:�7
0�-
#� 
	intensity���������
p

 
� "�����������
)__inference_model_1_layer_call_fn_1359752�:#$/,.-78?@KHJIST[\gdfeopwx����������������7�4
-�*
 �
inputs���������
p 

 
� "�����������
)__inference_model_1_layer_call_fn_1359841�:#$./,-78?@JKHIST[\fgdeopwx����������������7�4
-�*
 �
inputs���������
p

 
� "�����������
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1361263f����3�0
)�&
 �
inputs���������2
p 
� "%�"
�
0���������2
� �
G__inference_reg_norm_1_layer_call_and_return_conditional_losses_1361297f����3�0
)�&
 �
inputs���������2
p
� "%�"
�
0���������2
� �
,__inference_reg_norm_1_layer_call_fn_1361230Y����3�0
)�&
 �
inputs���������2
p 
� "����������2�
,__inference_reg_norm_1_layer_call_fn_1361243Y����3�0
)�&
 �
inputs���������2
p
� "����������2�
F__inference_regress_1_layer_call_and_return_conditional_losses_1361217^��/�,
%�"
 �
inputs���������
� "%�"
�
0���������2
� �
+__inference_regress_1_layer_call_fn_1361200Q��/�,
%�"
 �
inputs���������
� "����������2�
%__inference_signature_wrapper_1359633�:#$/,.-78?@KHJIST[\gdfeopwx����������������?�<
� 
5�2
0
	intensity#� 
	intensity���������"-�*
(
dense�
dense���������