ШЁ
+с*
.
Abs
x"T
y"T"
Ttype:

2	
Y
AddN
inputs"T*N
sum"T"
Nint(0"#
Ttype:
2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
$
DisableCopyOnRead
resource
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorMod
x"T
y"T
z"T"
Ttype:
2	
q
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
Ў
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
$

LogicalAnd
x

y

z

>
Maximum
x"T
y"T
z"T"
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
:
OnesLike
x"T
y"T"
Ttype:
2	

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
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Ѕ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeэout_type"	
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
1
Sign
x"T
y"T"
Ttype:
2
	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
:
SqrtGrad
y"T
dy"T
z"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
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

StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
t
TensorScatterAdd
tensor"T
indices"Tindices
updates"T
output"T"	
Ttype"
Tindicestype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
Ф
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T""
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
А
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.15.02v2.15.0-2-g0b15fdfcb3f8
N
ConstConst*
_output_shapes
: *
dtype0*
valueB 2Zй­фKм?
P
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2        
P
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2      №?
P
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2      Р?
P
Const_4Const*
_output_shapes
: *
dtype0*
valueB 2      №?
P
Const_5Const*
_output_shapes
: *
dtype0*
valueB 2      №?
Y
Const_6Const*
_output_shapes

:*
dtype0*
valueB: 
P
Const_7Const*
_output_shapes
: *
dtype0*
valueB 2Ь;f ц?
Q
Const_8Const*
_output_shapes
:*
dtype0*
valueB: 
P
Const_9Const*
_output_shapes
: *
dtype0*
valueB 2::20ЯІ?
Q
Const_10Const*
_output_shapes
: *
dtype0*
valueB 2Ь;f ц?
Q
Const_11Const*
_output_shapes
: *
dtype0*
valueB 2jяДј[@

Const_12Const*
_output_shapes
:*
dtype0*E
value<B:"0                        3EЇyтП               

Const_13Const*
_output_shapes
:*
dtype0*E
value<B:"0        ЊLXшzЖћ?.!	ѓПкNOБоћў?Јєwу@ЈєwуёП
Q
Const_14Const*
_output_shapes
: *
dtype0*
valueB 2-DTћ!	@
u
Const_15Const*
_output_shapes
:	*
dtype0*9
value0B.	"$                            
Q
Const_16Const*
_output_shapes
: *
dtype0*
valueB 2      Р?
Q
Const_17Const*
_output_shapes
: *
dtype0*
valueB 2      Р?
Q
Const_18Const*
_output_shapes
: *
dtype0*
valueB 2      Р?
Q
Const_19Const*
_output_shapes
: *
dtype0*
valueB 2Ь;f ц?
Q
Const_20Const*
_output_shapes
: *
dtype0*
valueB 23ЇЈе#іI9
е
"DenseLayer/DenseLayer_DenseLayer__VarHandleOp*
_output_shapes
: *3

debug_name%#DenseLayer/DenseLayer_DenseLayer__/*
dtype0*
shape
:@*3
shared_name$"DenseLayer/DenseLayer_DenseLayer__

6DenseLayer/DenseLayer_DenseLayer__/Read/ReadVariableOpReadVariableOp"DenseLayer/DenseLayer_DenseLayer__*
_output_shapes

:@*
dtype0
л
$DenseLayer/DenseLayer_DenseLayer___1VarHandleOp*
_output_shapes
: *5

debug_name'%DenseLayer/DenseLayer_DenseLayer___1/*
dtype0*
shape
:@*5
shared_name&$DenseLayer/DenseLayer_DenseLayer___1

8DenseLayer/DenseLayer_DenseLayer___1/Read/ReadVariableOpReadVariableOp$DenseLayer/DenseLayer_DenseLayer___1*
_output_shapes

:@*
dtype0
ъ
)DenseLayer/DenseLayer_DenseLayer_no_decayVarHandleOp*
_output_shapes
: *:

debug_name,*DenseLayer/DenseLayer_DenseLayer_no_decay/*
dtype0*
shape
:@*:
shared_name+)DenseLayer/DenseLayer_DenseLayer_no_decay
Ї
=DenseLayer/DenseLayer_DenseLayer_no_decay/Read/ReadVariableOpReadVariableOp)DenseLayer/DenseLayer_DenseLayer_no_decay*
_output_shapes

:@*
dtype0
№
+DenseLayer/DenseLayer_DenseLayer_no_decay_1VarHandleOp*
_output_shapes
: *<

debug_name.,DenseLayer/DenseLayer_DenseLayer_no_decay_1/*
dtype0*
shape
:@@*<
shared_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_1
Ћ
?DenseLayer/DenseLayer_DenseLayer_no_decay_1/Read/ReadVariableOpReadVariableOp+DenseLayer/DenseLayer_DenseLayer_no_decay_1*
_output_shapes

:@@*
dtype0
№
+DenseLayer/DenseLayer_DenseLayer_no_decay_2VarHandleOp*
_output_shapes
: *<

debug_name.,DenseLayer/DenseLayer_DenseLayer_no_decay_2/*
dtype0*
shape
:@@*<
shared_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_2
Ћ
?DenseLayer/DenseLayer_DenseLayer_no_decay_2/Read/ReadVariableOpReadVariableOp+DenseLayer/DenseLayer_DenseLayer_no_decay_2*
_output_shapes

:@@*
dtype0
№
+DenseLayer/DenseLayer_DenseLayer_no_decay_3VarHandleOp*
_output_shapes
: *<

debug_name.,DenseLayer/DenseLayer_DenseLayer_no_decay_3/*
dtype0*
shape
:@*<
shared_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_3
Ћ
?DenseLayer/DenseLayer_DenseLayer_no_decay_3/Read/ReadVariableOpReadVariableOp+DenseLayer/DenseLayer_DenseLayer_no_decay_3*
_output_shapes

:@*
dtype0
ч
(ChemIndTransf/DenseLayer_ChemIndTransf__VarHandleOp*
_output_shapes
: *9

debug_name+)ChemIndTransf/DenseLayer_ChemIndTransf__/*
dtype0*
shape
:*9
shared_name*(ChemIndTransf/DenseLayer_ChemIndTransf__
Ѕ
<ChemIndTransf/DenseLayer_ChemIndTransf__/Read/ReadVariableOpReadVariableOp(ChemIndTransf/DenseLayer_ChemIndTransf__*
_output_shapes

:*
dtype0
Ё
rho/reducing_AVarHandleOp*
_output_shapes
: *

debug_namerho/reducing_A/*
dtype0*
shape:*
shared_namerho/reducing_A
y
"rho/reducing_A/Read/ReadVariableOpReadVariableOprho/reducing_A*&
_output_shapes
:*
dtype0
Ј
Z/ChemicalEmbeddingVarHandleOp*
_output_shapes
: *$

debug_nameZ/ChemicalEmbedding/*
dtype0*
shape
:*$
shared_nameZ/ChemicalEmbedding
{
'Z/ChemicalEmbedding/Read/ReadVariableOpReadVariableOpZ/ChemicalEmbedding*
_output_shapes

:*
dtype0

element_map_indexVarHandleOp*
_output_shapes
: *"

debug_nameelement_map_index/*
dtype0*
shape:*"
shared_nameelement_map_index
s
%element_map_index/Read/ReadVariableOpReadVariableOpelement_map_index*
_output_shapes
:*
dtype0
Є
element_map_symbolsVarHandleOp*
_output_shapes
: *$

debug_nameelement_map_symbols/*
dtype0*
shape:*$
shared_nameelement_map_symbols
w
'element_map_symbols/Read/ReadVariableOpReadVariableOpelement_map_symbols*
_output_shapes
:*
dtype0


RBF_cutoffVarHandleOp*
_output_shapes
: *

debug_nameRBF_cutoff/*
dtype0*
shape
:*
shared_name
RBF_cutoff
i
RBF_cutoff/Read/ReadVariableOpReadVariableOp
RBF_cutoff*
_output_shapes

:*
dtype0
v
serving_default_atomic_mu_iPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
^
serving_default_batch_tot_natPlaceholder*
_output_shapes
: *
dtype0*
shape: 
c
"serving_default_batch_tot_nat_realPlaceholder*
_output_shapes
: *
dtype0*
shape: 
~
serving_default_bond_vectorPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
p
serving_default_ind_iPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
p
serving_default_ind_jPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
o
serving_default_mu_iPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
o
serving_default_mu_jPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
с
StatefulPartitionedCallStatefulPartitionedCallserving_default_atomic_mu_iserving_default_batch_tot_nat"serving_default_batch_tot_nat_realserving_default_bond_vectorserving_default_ind_iserving_default_ind_jserving_default_mu_iserving_default_mu_jConst_20
RBF_cutoff+DenseLayer/DenseLayer_DenseLayer_no_decay_3Const_19+DenseLayer/DenseLayer_DenseLayer_no_decay_2Const_18+DenseLayer/DenseLayer_DenseLayer_no_decay_1Const_17)DenseLayer/DenseLayer_DenseLayer_no_decayConst_16Const_15Const_14Const_13Const_12Const_11(ChemIndTransf/DenseLayer_ChemIndTransf__Const_10Z/ChemicalEmbeddingConst_9rho/reducing_AConst_8Const_7Const_6Const_5$DenseLayer/DenseLayer_DenseLayer___1Const_4"DenseLayer/DenseLayer_DenseLayer__Const_3Const_2Const_1Const*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:џџџџџџџџџ::џџџџџџџџџ:*,
_read_only_resource_inputs

	
 "*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_9812

NoOpNoOp
Д!
Const_21Const"/device:CPU:0*
_output_shapes
: *
dtype0*ь 
valueт Bп  Bи 
c
instructions
compute_specs
train_specs

slices
compute

signatures*
R
0
1
	2

3
4
5
6
7
8
9
10*
y
	ind_i
	ind_j
bond_vector
mu_i
mu_j
batch_tot_nat
atomic_mu_i
batch_tot_nat_real* 
Љ
	ind_i
	ind_j
map_atoms_to_structure
n_struct_total
bond_vector
batch_tot_nat
 mu_i
!mu_j
"atomic_mu_i
#batch_tot_nat_real* 
* 

$trace_0* 

%serving_default* 

&
_init_args* 

'
_init_args* 
:
(
_init_args
)cutoff_dict
*bond_cutoff_map*
0
+
_init_args
,hidden_layers
-mlp*

.
_init_args
/sg* 
K
0
_init_args
1element_map_symbols
2element_map_index
3w*
O
4
_init_args


radial
angular
	indicator
5lin_transform*
|
6
_init_args
7instructions

8ls_max
9allowed_l_p
:	collector
;downscale_embeddings
<
reducing_A*

=
_init_args* 
H
>
_init_args

target

?origin
@hidden_layers
Amlp*

B
_init_args

target* 

	Cshape* 

	Dshape* 

	Eshape* 

	Fshape* 

	Gshape* 

	Hshape* 

	Ishape* 

	Jshape* 

	Cshape* 

	Kshape* 

	Lshape* 

	Mshape* 

	Eshape* 

	Hshape* 

	Fshape* 

	Gshape* 

	Ishape* 

	Jshape* 
Э
N	capture_0
O	capture_3
P	capture_5
Q	capture_7
R	capture_9
S
capture_10
T
capture_11
U
capture_12
V
capture_13
W
capture_14
X
capture_16
Y
capture_18
Z
capture_20
[
capture_21
\
capture_22
]
capture_23
^
capture_25
_
capture_27
`
capture_28
a
capture_29
b
capture_30* 
Э
N	capture_0
O	capture_3
P	capture_5
Q	capture_7
R	capture_9
S
capture_10
T
capture_11
U
capture_12
V
capture_13
W
capture_14
X
capture_16
Y
capture_18
Z
capture_20
[
capture_21
\
capture_22
]
capture_23
^
capture_25
_
capture_27
`
capture_28
a
capture_29
b
capture_30* 
* 

bond_length* 
/
	bonds
celement_map
dcutoff_dict* 
* 
]W
VARIABLE_VALUE
RBF_cutoff9instructions/2/bond_cutoff_map/.ATTRIBUTES/VARIABLE_VALUE*

		basis*
* 
G
elayers_config

flayer0

glayer1

hlayer2

ilayer3*

vhat* 
* 

jelement_map* 
jd
VARIABLE_VALUEelement_map_symbols=instructions/5/element_map_symbols/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEelement_map_index;instructions/5/element_map_index/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEZ/ChemicalEmbedding+instructions/5/w/.ATTRIBUTES/VARIABLE_VALUE*
,
	indicator


radial
angular*

kw*
3
linstructions

mls_max
nallowed_l_p*

0*
* 
	
o0* 
	
pA* 
* 
\V
VARIABLE_VALUErho/reducing_A4instructions/7/reducing_A/.ATTRIBUTES/VARIABLE_VALUE*
* 
/
qhidden_layers

rorigin

target*

0*
* 
/
slayers_config

tlayer0

ulayer1*


target* 
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
* 
* 
* 

vw*

ww*

xw*

yw*
* 
{u
VARIABLE_VALUE(ChemIndTransf/DenseLayer_ChemIndTransf__9instructions/6/lin_transform/w/.ATTRIBUTES/VARIABLE_VALUE*

0*
* 
	
o0* 
* 
* 
* 

0*
* 

zw*

{w*
{u
VARIABLE_VALUE+DenseLayer/DenseLayer_DenseLayer_no_decay_36instructions/3/mlp/layer0/w/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE+DenseLayer/DenseLayer_DenseLayer_no_decay_26instructions/3/mlp/layer1/w/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE+DenseLayer/DenseLayer_DenseLayer_no_decay_16instructions/3/mlp/layer2/w/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE)DenseLayer/DenseLayer_DenseLayer_no_decay6instructions/3/mlp/layer3/w/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE$DenseLayer/DenseLayer_DenseLayer___16instructions/9/mlp/layer0/w/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE"DenseLayer/DenseLayer_DenseLayer__6instructions/9/mlp/layer1/w/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ј
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename
RBF_cutoffelement_map_symbolselement_map_indexZ/ChemicalEmbeddingrho/reducing_A(ChemIndTransf/DenseLayer_ChemIndTransf__+DenseLayer/DenseLayer_DenseLayer_no_decay_3+DenseLayer/DenseLayer_DenseLayer_no_decay_2+DenseLayer/DenseLayer_DenseLayer_no_decay_1)DenseLayer/DenseLayer_DenseLayer_no_decay$DenseLayer/DenseLayer_DenseLayer___1"DenseLayer/DenseLayer_DenseLayer__Const_21*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_10045
 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
RBF_cutoffelement_map_symbolselement_map_indexZ/ChemicalEmbeddingrho/reducing_A(ChemIndTransf/DenseLayer_ChemIndTransf__+DenseLayer/DenseLayer_DenseLayer_no_decay_3+DenseLayer/DenseLayer_DenseLayer_no_decay_2+DenseLayer/DenseLayer_DenseLayer_no_decay_1)DenseLayer/DenseLayer_DenseLayer_no_decay$DenseLayer/DenseLayer_DenseLayer___1"DenseLayer/DenseLayer_DenseLayer__*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_10090Ќ
О
И
!__inference_internal_grad_fn_9897
result_grads_0
result_grads_1
result_grads_2
mul_denselayer_cast 
mul_denselayer_einsum_einsum
identity

identity_1
mulMulmul_denselayer_castmul_denselayer_einsum_einsum^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@q
mul_1Mulmul_denselayer_castmul_denselayer_einsum_einsum*
T0*'
_output_shapes
:џџџџџџџџџ@N
sub/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
add/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@`
SquareSquaremul_denselayer_einsum_einsum*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?]
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:a]
'
_output_shapes
:џџџџџџџџџ@
2
_user_specified_nameDenseLayer/einsum/Einsum:GC

_output_shapes
: 
)
_user_specified_nameDenseLayer/Cast:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
а
М
!__inference_internal_grad_fn_9951
result_grads_0
result_grads_1
result_grads_2
mul_denselayer_cast_2"
mul_denselayer_einsum_2_einsum
identity

identity_1
mulMulmul_denselayer_cast_2mul_denselayer_einsum_2_einsum^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@u
mul_1Mulmul_denselayer_cast_2mul_denselayer_einsum_2_einsum*
T0*'
_output_shapes
:џџџџџџџџџ@N
sub/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
add/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@b
SquareSquaremul_denselayer_einsum_2_einsum*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?]
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:c_
'
_output_shapes
:џџџџџџџџџ@
4
_user_specified_nameDenseLayer/einsum_2/Einsum:IE

_output_shapes
: 
+
_user_specified_nameDenseLayer/Cast_2:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
@
џ
!__inference__traced_restore_10090
file_prefix-
assignvariableop_rbf_cutoff:4
&assignvariableop_1_element_map_symbols:2
$assignvariableop_2_element_map_index:8
&assignvariableop_3_z_chemicalembedding:;
!assignvariableop_4_rho_reducing_a:M
;assignvariableop_5_chemindtransf_denselayer_chemindtransf__:P
>assignvariableop_6_denselayer_denselayer_denselayer_no_decay_3:@P
>assignvariableop_7_denselayer_denselayer_denselayer_no_decay_2:@@P
>assignvariableop_8_denselayer_denselayer_denselayer_no_decay_1:@@N
<assignvariableop_9_denselayer_denselayer_denselayer_no_decay:@J
8assignvariableop_10_denselayer_denselayer_denselayer___1:@H
6assignvariableop_11_denselayer_denselayer_denselayer__:@
identity_13ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9В
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*и
valueЮBЫB9instructions/2/bond_cutoff_map/.ATTRIBUTES/VARIABLE_VALUEB=instructions/5/element_map_symbols/.ATTRIBUTES/VARIABLE_VALUEB;instructions/5/element_map_index/.ATTRIBUTES/VARIABLE_VALUEB+instructions/5/w/.ATTRIBUTES/VARIABLE_VALUEB4instructions/7/reducing_A/.ATTRIBUTES/VARIABLE_VALUEB9instructions/6/lin_transform/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer0/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer1/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer2/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer3/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/9/mlp/layer0/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/9/mlp/layer1/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B п
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOpAssignVariableOpassignvariableop_rbf_cutoffIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_1AssignVariableOp&assignvariableop_1_element_map_symbolsIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_2AssignVariableOp$assignvariableop_2_element_map_indexIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_3AssignVariableOp&assignvariableop_3_z_chemicalembeddingIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_4AssignVariableOp!assignvariableop_4_rho_reducing_aIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_5AssignVariableOp;assignvariableop_5_chemindtransf_denselayer_chemindtransf__Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_6AssignVariableOp>assignvariableop_6_denselayer_denselayer_denselayer_no_decay_3Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_7AssignVariableOp>assignvariableop_7_denselayer_denselayer_denselayer_no_decay_2Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_8AssignVariableOp>assignvariableop_8_denselayer_denselayer_denselayer_no_decay_1Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_9AssignVariableOp<assignvariableop_9_denselayer_denselayer_denselayer_no_decayIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_10AssignVariableOp8assignvariableop_10_denselayer_denselayer_denselayer___1Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_11AssignVariableOp6assignvariableop_11_denselayer_denselayer_denselayer__Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 з
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
:  
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_13Identity_13:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:B>
<
_user_specified_name$"DenseLayer/DenseLayer_DenseLayer__:D@
>
_user_specified_name&$DenseLayer/DenseLayer_DenseLayer___1:I
E
C
_user_specified_name+)DenseLayer/DenseLayer_DenseLayer_no_decay:K	G
E
_user_specified_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_1:KG
E
_user_specified_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_2:KG
E
_user_specified_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_3:HD
B
_user_specified_name*(ChemIndTransf/DenseLayer_ChemIndTransf__:.*
(
_user_specified_namerho/reducing_A:3/
-
_user_specified_nameZ/ChemicalEmbedding:1-
+
_user_specified_nameelement_map_index:3/
-
_user_specified_nameelement_map_symbols:*&
$
_user_specified_name
RBF_cutoff:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
а
М
!__inference_internal_grad_fn_9978
result_grads_0
result_grads_1
result_grads_2
mul_denselayer_cast_3"
mul_denselayer_einsum_4_einsum
identity

identity_1
mulMulmul_denselayer_cast_3mul_denselayer_einsum_4_einsum^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@u
mul_1Mulmul_denselayer_cast_3mul_denselayer_einsum_4_einsum*
T0*'
_output_shapes
:џџџџџџџџџ@N
sub/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
add/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@b
SquareSquaremul_denselayer_einsum_4_einsum*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?]
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:c_
'
_output_shapes
:џџџџџџџџџ@
4
_user_specified_nameDenseLayer/einsum_4/Einsum:IE

_output_shapes
: 
+
_user_specified_nameDenseLayer/Cast_3:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0


__inference_compute_9731
atomic_mu_i
batch_tot_nat
batch_tot_nat_real
bond_vector	
ind_i	
ind_j
mu_i
mu_j
scaledbondvector_add_yA
/bondspecificradialbasisfunction_gather_resource:4
"denselayer_readvariableop_resource:@
denselayer_mul_y6
$denselayer_readvariableop_1_resource:@@
denselayer_mul_4_y6
$denselayer_readvariableop_2_resource:@@
denselayer_mul_8_y6
$denselayer_readvariableop_3_resource:@
denselayer_mul_12_y
r_gatherv2_indices
y_mul_y

y_8392

y_8399

y_mul_35_y7
%chemindtransf_readvariableop_resource:
chemindtransf_mul_yE
3chemindtransf_einsum_einsum_readvariableop_resource:
a_mul_y@
&rho_gatherv2_1_readvariableop_resource:
rho_gatherv2_1_indices
	rho_mul_y 
rho_tensorscatteradd_indices
rho_mul_1_y6
$denselayer_readvariableop_4_resource:@
denselayer_mul_13_y6
$denselayer_readvariableop_5_resource:@
denselayer_mul_17_y	
mul_y
add_3_x
mul_2_y
identity

identity_1

identity_2

identity_3Ђ&BondSpecificRadialBasisFunction/GatherЂChemIndTransf/ReadVariableOpЂ*ChemIndTransf/einsum/Einsum/ReadVariableOpЂDenseLayer/ReadVariableOpЂDenseLayer/ReadVariableOp_1ЂDenseLayer/ReadVariableOp_2ЂDenseLayer/ReadVariableOp_3ЂDenseLayer/ReadVariableOp_4ЂDenseLayer/ReadVariableOp_5Ђrho/GatherV2_1/ReadVariableOpf
BondLength/norm/mulMulbond_vectorbond_vector*
T0*'
_output_shapes
:џџџџџџџџџo
%BondLength/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:І
BondLength/norm/SumSumBondLength/norm/mul:z:0.BondLength/norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(l
BondLength/norm/SqrtSqrtBondLength/norm/Sum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
ScaledBondVector/addAddV2BondLength/norm/Sqrt:y:0scaledbondvector_add_y*
T0*'
_output_shapes
:џџџџџџџџџ|
ScaledBondVector/truedivRealDivbond_vectorScaledBondVector/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџg
%BondSpecificRadialBasisFunction/ConstConst*
_output_shapes
: *
dtype0*
value	B :
#BondSpecificRadialBasisFunction/mulMulmu_i.BondSpecificRadialBasisFunction/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
#BondSpecificRadialBasisFunction/addAddV2mu_j'BondSpecificRadialBasisFunction/mul:z:0*
T0*#
_output_shapes
:џџџџџџџџџи
&BondSpecificRadialBasisFunction/GatherResourceGather/bondspecificradialbasisfunction_gather_resource'BondSpecificRadialBasisFunction/add:z:0*
Tindices0*'
_output_shapes
:џџџџџџџџџ*
dtype0Џ
'BondSpecificRadialBasisFunction/truedivRealDivBondLength/norm/Sqrt:y:0/BondSpecificRadialBasisFunction/Gather:output:0*
T0*'
_output_shapes
:џџџџџџџџџn
%BondSpecificRadialBasisFunction/sub/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Й
#BondSpecificRadialBasisFunction/subSub.BondSpecificRadialBasisFunction/sub/x:output:0+BondSpecificRadialBasisFunction/truediv:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
#BondSpecificRadialBasisFunction/AbsAbs'BondSpecificRadialBasisFunction/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџp
'BondSpecificRadialBasisFunction/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Й
%BondSpecificRadialBasisFunction/sub_1Sub0BondSpecificRadialBasisFunction/sub_1/x:output:0'BondSpecificRadialBasisFunction/Abs:y:0*
T0*'
_output_shapes
:џџџџџџџџџp
'BondSpecificRadialBasisFunction/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB 2       @Л
%BondSpecificRadialBasisFunction/mul_1Mul0BondSpecificRadialBasisFunction/mul_1/x:output:0)BondSpecificRadialBasisFunction/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџp
'BondSpecificRadialBasisFunction/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB 2      №?Л
%BondSpecificRadialBasisFunction/sub_2Sub)BondSpecificRadialBasisFunction/mul_1:z:00BondSpecificRadialBasisFunction/sub_2/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
)BondSpecificRadialBasisFunction/ones_likeOnesLike)BondSpecificRadialBasisFunction/sub_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџp
'BondSpecificRadialBasisFunction/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB 2       @Л
%BondSpecificRadialBasisFunction/mul_2Mul0BondSpecificRadialBasisFunction/mul_2/x:output:0)BondSpecificRadialBasisFunction/sub_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџД
%BondSpecificRadialBasisFunction/mul_3Mul)BondSpecificRadialBasisFunction/sub_2:z:0)BondSpecificRadialBasisFunction/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџИ
%BondSpecificRadialBasisFunction/sub_3Sub)BondSpecificRadialBasisFunction/mul_3:z:0-BondSpecificRadialBasisFunction/ones_like:y:0*
T0*'
_output_shapes
:џџџџџџџџџё
%BondSpecificRadialBasisFunction/stackPack-BondSpecificRadialBasisFunction/ones_like:y:0)BondSpecificRadialBasisFunction/sub_2:z:0)BondSpecificRadialBasisFunction/sub_3:z:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ
.BondSpecificRadialBasisFunction/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          е
)BondSpecificRadialBasisFunction/transpose	Transpose.BondSpecificRadialBasisFunction/stack:output:07BondSpecificRadialBasisFunction/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
3BondSpecificRadialBasisFunction/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
5BondSpecificRadialBasisFunction/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
5BondSpecificRadialBasisFunction/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ѓ
-BondSpecificRadialBasisFunction/strided_sliceStridedSlice-BondSpecificRadialBasisFunction/transpose:y:0<BondSpecificRadialBasisFunction/strided_slice/stack:output:0>BondSpecificRadialBasisFunction/strided_slice/stack_1:output:0>BondSpecificRadialBasisFunction/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask
5BondSpecificRadialBasisFunction/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
7BondSpecificRadialBasisFunction/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
7BondSpecificRadialBasisFunction/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
/BondSpecificRadialBasisFunction/strided_slice_1StridedSlice6BondSpecificRadialBasisFunction/strided_slice:output:0>BondSpecificRadialBasisFunction/strided_slice_1/stack:output:0@BondSpecificRadialBasisFunction/strided_slice_1/stack_1:output:0@BondSpecificRadialBasisFunction/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskn
%BondSpecificRadialBasisFunction/pow/yConst*
_output_shapes
: *
dtype0*
valueB 2      @Й
#BondSpecificRadialBasisFunction/powPow+BondSpecificRadialBasisFunction/truediv:z:0.BondSpecificRadialBasisFunction/pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџp
'BondSpecificRadialBasisFunction/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB 2      5@Й
%BondSpecificRadialBasisFunction/mul_4Mul0BondSpecificRadialBasisFunction/mul_4/x:output:0'BondSpecificRadialBasisFunction/pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџp
'BondSpecificRadialBasisFunction/sub_4/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Л
%BondSpecificRadialBasisFunction/sub_4Sub0BondSpecificRadialBasisFunction/sub_4/x:output:0)BondSpecificRadialBasisFunction/mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџp
'BondSpecificRadialBasisFunction/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB 2      @Н
%BondSpecificRadialBasisFunction/pow_1Pow+BondSpecificRadialBasisFunction/truediv:z:00BondSpecificRadialBasisFunction/pow_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџp
'BondSpecificRadialBasisFunction/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB 2     A@Л
%BondSpecificRadialBasisFunction/mul_5Mul0BondSpecificRadialBasisFunction/mul_5/x:output:0)BondSpecificRadialBasisFunction/pow_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџЖ
%BondSpecificRadialBasisFunction/add_1AddV2)BondSpecificRadialBasisFunction/sub_4:z:0)BondSpecificRadialBasisFunction/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџp
'BondSpecificRadialBasisFunction/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB 2      @Н
%BondSpecificRadialBasisFunction/pow_2Pow+BondSpecificRadialBasisFunction/truediv:z:00BondSpecificRadialBasisFunction/pow_2/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџp
'BondSpecificRadialBasisFunction/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB 2      .@Л
%BondSpecificRadialBasisFunction/mul_6Mul0BondSpecificRadialBasisFunction/mul_6/x:output:0)BondSpecificRadialBasisFunction/pow_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџД
%BondSpecificRadialBasisFunction/sub_5Sub)BondSpecificRadialBasisFunction/add_1:z:0)BondSpecificRadialBasisFunction/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџУ
%BondSpecificRadialBasisFunction/mul_7Mul8BondSpecificRadialBasisFunction/strided_slice_1:output:0)BondSpecificRadialBasisFunction/sub_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџЙ
,BondSpecificRadialBasisFunction/GreaterEqualGreaterEqualBondLength/norm/Sqrt:y:0/BondSpecificRadialBasisFunction/Gather:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
*BondSpecificRadialBasisFunction/zeros_like	ZerosLike)BondSpecificRadialBasisFunction/mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџѓ
(BondSpecificRadialBasisFunction/SelectV2SelectV20BondSpecificRadialBasisFunction/GreaterEqual:z:0.BondSpecificRadialBasisFunction/zeros_like:y:0)BondSpecificRadialBasisFunction/mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ|
DenseLayer/ReadVariableOpReadVariableOp"denselayer_readvariableop_resource*
_output_shapes

:@*
dtype0s
DenseLayer/mulMul!DenseLayer/ReadVariableOp:value:0denselayer_mul_y*
T0*
_output_shapes

:@Ф
DenseLayer/einsum/EinsumEinsum1BondSpecificRadialBasisFunction/SelectV2:output:0DenseLayer/mul:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@*
equation...k,...kn->...nT
DenseLayer/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
DenseLayer/CastCastDenseLayer/beta:output:0*

DstT0*

SrcT0*
_output_shapes
: 
DenseLayer/mul_1MulDenseLayer/Cast:y:0!DenseLayer/einsum/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@e
DenseLayer/SigmoidSigmoidDenseLayer/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
DenseLayer/mul_2Mul!DenseLayer/einsum/Einsum:output:0DenseLayer/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@g
DenseLayer/IdentityIdentityDenseLayer/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@ч
DenseLayer/IdentityN	IdentityNDenseLayer/mul_2:z:0!DenseLayer/einsum/Einsum:output:0DenseLayer/Cast:y:0*
T
2**
_gradient_op_typeCustomGradient-8301*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: [
DenseLayer/mul_3/yConst*
_output_shapes
: *
dtype0*
valueB 2ЦмЕ|ањ?
DenseLayer/mul_3MulDenseLayer/IdentityN:output:0DenseLayer/mul_3/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
DenseLayer/ReadVariableOp_1ReadVariableOp$denselayer_readvariableop_1_resource*
_output_shapes

:@@*
dtype0y
DenseLayer/mul_4Mul#DenseLayer/ReadVariableOp_1:value:0denselayer_mul_4_y*
T0*
_output_shapes

:@@Ћ
DenseLayer/einsum_1/EinsumEinsumDenseLayer/mul_3:z:0DenseLayer/mul_4:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@*
equation...k,...kn->...nV
DenseLayer/beta_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ?e
DenseLayer/Cast_1CastDenseLayer/beta_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 
DenseLayer/mul_5MulDenseLayer/Cast_1:y:0#DenseLayer/einsum_1/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@g
DenseLayer/Sigmoid_1SigmoidDenseLayer/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
DenseLayer/mul_6Mul#DenseLayer/einsum_1/Einsum:output:0DenseLayer/Sigmoid_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@i
DenseLayer/Identity_1IdentityDenseLayer/mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@э
DenseLayer/IdentityN_1	IdentityNDenseLayer/mul_6:z:0#DenseLayer/einsum_1/Einsum:output:0DenseLayer/Cast_1:y:0*
T
2**
_gradient_op_typeCustomGradient-8318*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: [
DenseLayer/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB 2ЦмЕ|ањ?
DenseLayer/mul_7MulDenseLayer/IdentityN_1:output:0DenseLayer/mul_7/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
DenseLayer/ReadVariableOp_2ReadVariableOp$denselayer_readvariableop_2_resource*
_output_shapes

:@@*
dtype0y
DenseLayer/mul_8Mul#DenseLayer/ReadVariableOp_2:value:0denselayer_mul_8_y*
T0*
_output_shapes

:@@Ћ
DenseLayer/einsum_2/EinsumEinsumDenseLayer/mul_7:z:0DenseLayer/mul_8:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@*
equation...k,...kn->...nV
DenseLayer/beta_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ?e
DenseLayer/Cast_2CastDenseLayer/beta_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 
DenseLayer/mul_9MulDenseLayer/Cast_2:y:0#DenseLayer/einsum_2/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@g
DenseLayer/Sigmoid_2SigmoidDenseLayer/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
DenseLayer/mul_10Mul#DenseLayer/einsum_2/Einsum:output:0DenseLayer/Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@j
DenseLayer/Identity_2IdentityDenseLayer/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@ю
DenseLayer/IdentityN_2	IdentityNDenseLayer/mul_10:z:0#DenseLayer/einsum_2/Einsum:output:0DenseLayer/Cast_2:y:0*
T
2**
_gradient_op_typeCustomGradient-8335*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \
DenseLayer/mul_11/yConst*
_output_shapes
: *
dtype0*
valueB 2ЦмЕ|ањ?
DenseLayer/mul_11MulDenseLayer/IdentityN_2:output:0DenseLayer/mul_11/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
DenseLayer/ReadVariableOp_3ReadVariableOp$denselayer_readvariableop_3_resource*
_output_shapes

:@*
dtype0{
DenseLayer/mul_12Mul#DenseLayer/ReadVariableOp_3:value:0denselayer_mul_12_y*
T0*
_output_shapes

:@­
DenseLayer/einsum_3/EinsumEinsumDenseLayer/mul_11:z:0DenseLayer/mul_12:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ*
equation...k,...kn->...nd
R/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      
	R/ReshapeReshape#DenseLayer/einsum_3/Einsum:output:0R/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ
R/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџБ

R/GatherV2GatherV2R/Reshape:output:0r_gatherv2_indicesR/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:џџџџџџџџџ	f
Y/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
Y/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
Y/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Y/strided_sliceStridedSliceScaledBondVector/truediv:z:0Y/strided_slice/stack:output:0 Y/strided_slice/stack_1:output:0 Y/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskh
Y/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       j
Y/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       j
Y/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Y/strided_slice_1StridedSliceScaledBondVector/truediv:z:0 Y/strided_slice_1/stack:output:0"Y/strided_slice_1/stack_1:output:0"Y/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskh
Y/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       j
Y/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       j
Y/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Y/strided_slice_2StridedSliceScaledBondVector/truediv:z:0 Y/strided_slice_2/stack:output:0"Y/strided_slice_2/stack_1:output:0"Y/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskP
Y/mul/xConst*
_output_shapes
: *
dtype0*
valueB 2      @H
Y/mulMulY/mul/x:output:0y_mul_y*
T0*
_output_shapes
: T
Y/truediv/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?V
	Y/truedivRealDivY/truediv/x:output:0	Y/mul:z:0*
T0*
_output_shapes
: >
Y/SqrtSqrtY/truediv:z:0*
T0*
_output_shapes
: R
	Y/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?O
Y/mul_1MulY/mul_1/x:output:0
Y/Sqrt:y:0*
T0*
_output_shapes
: R
	Y/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB 2        l
Y/mul_2MulY/strided_slice_2:output:0Y/mul_2/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџV
Y/addAddV2Y/mul_2:z:0Y/mul_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџR
	Y/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB 2      @L
Y/mul_3MulY/mul_3/x:output:0y_mul_y*
T0*
_output_shapes
: V
Y/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      @\
Y/truediv_1RealDivY/truediv_1/x:output:0Y/mul_3:z:0*
T0*
_output_shapes
: B
Y/Sqrt_1SqrtY/truediv_1:z:0*
T0*
_output_shapes
: R
	Y/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB 2       @L
Y/mul_4MulY/mul_4/x:output:0y_mul_y*
T0*
_output_shapes
: V
Y/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB 2      @\
Y/truediv_2RealDivY/truediv_2/x:output:0Y/mul_4:z:0*
T0*
_output_shapes
: B
Y/Sqrt_2SqrtY/truediv_2:z:0*
T0*
_output_shapes
: f
Y/mul_5MulY/Sqrt_1:y:0Y/strided_slice_2:output:0*
T0*#
_output_shapes
:џџџџџџџџџR
	Y/mul_6/yConst*
_output_shapes
: *
dtype0*
valueB 2        l
Y/mul_6MulY/strided_slice_2:output:0Y/mul_6/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџU
Y/subSubY/mul_6:z:0Y/Sqrt_2:y:0*
T0*#
_output_shapes
:џџџџџџџџџa
Y/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:c
Y/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
Y/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
Y/strided_slice_3StridedSlicey_8392 Y/strided_slice_3/stack:output:0"Y/strided_slice_3/stack_1:output:0"Y/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
Y/mul_7MulY/strided_slice_2:output:0Y/mul_5:z:0*
T0*#
_output_shapes
:џџџџџџџџџa
Y/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:c
Y/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
Y/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
Y/strided_slice_4StridedSlicey_8399 Y/strided_slice_4/stack:output:0"Y/strided_slice_4/stack_1:output:0"Y/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
Y/mul_8MulY/strided_slice_4:output:0	Y/add:z:0*
T0*#
_output_shapes
:џџџџџџџџџX
Y/add_1AddV2Y/mul_7:z:0Y/mul_8:z:0*
T0*#
_output_shapes
:џџџџџџџџџe
Y/mul_9MulY/strided_slice_3:output:0Y/add_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџP
Y/ConstConst*
_output_shapes
: *
dtype0*
valueB 2      @R
	Y/add_2/xConst*
_output_shapes
: *
dtype0*
valueB 2       @W
Y/add_2AddV2Y/add_2/x:output:0Y/Const:output:0*
T0*
_output_shapes
: >
Y/Sqrt_3SqrtY/add_2:z:0*
T0*
_output_shapes
: g
Y/mul_10MulY/strided_slice_2:output:0Y/Sqrt_3:y:0*
T0*#
_output_shapes
:џџџџџџџџџV
Y/mul_11MulY/mul_10:z:0	Y/sub:z:0*
T0*#
_output_shapes
:џџџџџџџџџa
Y/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:c
Y/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
Y/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
Y/strided_slice_5StridedSlicey_8392 Y/strided_slice_5/stack:output:0"Y/strided_slice_5/stack_1:output:0"Y/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
Y/mul_12MulY/strided_slice_5:output:0	Y/sub:z:0*
T0*#
_output_shapes
:џџџџџџџџџ
Y/stackPack	Y/add:z:0Y/mul_5:z:0	Y/sub:z:0Y/mul_9:z:0Y/mul_11:z:0Y/mul_12:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџa
Y/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: c
Y/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
Y/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:№
Y/strided_slice_6StridedSliceY/stack:output:0 Y/strided_slice_6/stack:output:0"Y/strided_slice_6/stack_1:output:0"Y/strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_maska
Y/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB: c
Y/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
Y/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:№
Y/strided_slice_7StridedSliceY/stack:output:0 Y/strided_slice_7/stack:output:0"Y/strided_slice_7/stack_1:output:0"Y/strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskc
Y/zeros_like	ZerosLikeY/strided_slice_7:output:0*
T0*#
_output_shapes
:џџџџџџџџџa
Y/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:c
Y/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
Y/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:№
Y/strided_slice_8StridedSliceY/stack:output:0 Y/strided_slice_8/stack:output:0"Y/strided_slice_8/stack_1:output:0"Y/strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_maska
Y/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:c
Y/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
Y/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:№
Y/strided_slice_9StridedSliceY/stack:output:0 Y/strided_slice_9/stack:output:0"Y/strided_slice_9/stack_1:output:0"Y/strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
Y/zeros_like_1	ZerosLikeY/strided_slice_9:output:0*
T0*#
_output_shapes
:џџџџџџџџџb
Y/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
Y/strided_slice_10StridedSliceY/stack:output:0!Y/strided_slice_10/stack:output:0#Y/strided_slice_10/stack_1:output:0#Y/strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskb
Y/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
Y/strided_slice_11StridedSliceY/stack:output:0!Y/strided_slice_11/stack:output:0#Y/strided_slice_11/stack_1:output:0#Y/strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskf
Y/zeros_like_2	ZerosLikeY/strided_slice_11:output:0*
T0*#
_output_shapes
:џџџџџџџџџb
Y/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
Y/strided_slice_12StridedSliceY/stack:output:0!Y/strided_slice_12/stack:output:0#Y/strided_slice_12/stack_1:output:0#Y/strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskt
Y/mul_13MulY/strided_slice:output:0Y/strided_slice_12:output:0*
T0*#
_output_shapes
:џџџџџџџџџb
Y/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
Y/strided_slice_13StridedSliceY/stack:output:0!Y/strided_slice_13/stack:output:0#Y/strided_slice_13/stack_1:output:0#Y/strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskv
Y/mul_14MulY/strided_slice_1:output:0Y/strided_slice_13:output:0*
T0*#
_output_shapes
:џџџџџџџџџb
Y/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
Y/strided_slice_14StridedSliceY/stack:output:0!Y/strided_slice_14/stack:output:0#Y/strided_slice_14/stack_1:output:0#Y/strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskt
Y/mul_15MulY/strided_slice:output:0Y/strided_slice_14:output:0*
T0*#
_output_shapes
:џџџџџџџџџb
Y/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
Y/strided_slice_15StridedSliceY/stack:output:0!Y/strided_slice_15/stack:output:0#Y/strided_slice_15/stack_1:output:0#Y/strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskv
Y/mul_16MulY/strided_slice_1:output:0Y/strided_slice_15:output:0*
T0*#
_output_shapes
:џџџџџџџџџq
Y/mul_17MulY/strided_slice:output:0Y/strided_slice:output:0*
T0*#
_output_shapes
:џџџџџџџџџu
Y/mul_18MulY/strided_slice_1:output:0Y/strided_slice_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџX
Y/sub_1SubY/mul_17:z:0Y/mul_18:z:0*
T0*#
_output_shapes
:џџџџџџџџџs
Y/mul_19MulY/strided_slice:output:0Y/strided_slice_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџs
Y/mul_20MulY/strided_slice_1:output:0Y/strided_slice:output:0*
T0*#
_output_shapes
:џџџџџџџџџZ
Y/add_3AddV2Y/mul_19:z:0Y/mul_20:z:0*
T0*#
_output_shapes
:џџџџџџџџџb
Y/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
Y/strided_slice_16StridedSliceY/stack:output:0!Y/strided_slice_16/stack:output:0#Y/strided_slice_16/stack_1:output:0#Y/strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskg
Y/mul_21MulY/sub_1:z:0Y/strided_slice_16:output:0*
T0*#
_output_shapes
:џџџџџџџџџb
Y/strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
Y/strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
Y/strided_slice_17StridedSliceY/stack:output:0!Y/strided_slice_17/stack:output:0#Y/strided_slice_17/stack_1:output:0#Y/strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskg
Y/mul_22MulY/add_3:z:0Y/strided_slice_17:output:0*
T0*#
_output_shapes
:џџџџџџџџџI
Y/Abs/xConst*
_output_shapes
: *
dtype0*
value	B : ?
Y/AbsAbsY/Abs/x:output:0*
T0*
_output_shapes
: I
Y/mod/yConst*
_output_shapes
: *
dtype0*
value	B :O
Y/modFloorMod	Y/Abs:y:0Y/mod/y:output:0*
T0*
_output_shapes
: K
	Y/Equal/yConst*
_output_shapes
: *
dtype0*
value	B : P
Y/EqualEqual	Y/mod:z:0Y/Equal/y:output:0*
T0*
_output_shapes
: R
	Y/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2      №?R
	Y/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2      №Пl

Y/SelectV2SelectV2Y/Equal:z:0Y/Const_1:output:0Y/Const_2:output:0*
T0*
_output_shapes
: R
	Y/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2       @E
Y/Sqrt_4SqrtY/Const_3:output:0*
T0*
_output_shapes
: T
	Y/Abs_1/xConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџC
Y/Abs_1AbsY/Abs_1/x:output:0*
T0*
_output_shapes
: K
	Y/mod_1/yConst*
_output_shapes
: *
dtype0*
value	B :U
Y/mod_1FloorModY/Abs_1:y:0Y/mod_1/y:output:0*
T0*
_output_shapes
: M
Y/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : V
	Y/Equal_1EqualY/mod_1:z:0Y/Equal_1/y:output:0*
T0*
_output_shapes
: R
	Y/Const_4Const*
_output_shapes
: *
dtype0*
valueB 2      №?R
	Y/Const_5Const*
_output_shapes
: *
dtype0*
valueB 2      №Пp
Y/SelectV2_1SelectV2Y/Equal_1:z:0Y/Const_4:output:0Y/Const_5:output:0*
T0*
_output_shapes
: R
	Y/Const_6Const*
_output_shapes
: *
dtype0*
valueB 2       @E
Y/Sqrt_5SqrtY/Const_6:output:0*
T0*
_output_shapes
: Y
Y/mul_23MulY/Sqrt_5:y:0Y/mul_14:z:0*
T0*#
_output_shapes
:џџџџџџџџџb
Y/mul_24MulY/mul_23:z:0Y/SelectV2_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
	Y/Abs_2/xConst*
_output_shapes
: *
dtype0*
value	B : C
Y/Abs_2AbsY/Abs_2/x:output:0*
T0*
_output_shapes
: K
	Y/mod_2/yConst*
_output_shapes
: *
dtype0*
value	B :U
Y/mod_2FloorModY/Abs_2:y:0Y/mod_2/y:output:0*
T0*
_output_shapes
: M
Y/Equal_2/yConst*
_output_shapes
: *
dtype0*
value	B : V
	Y/Equal_2EqualY/mod_2:z:0Y/Equal_2/y:output:0*
T0*
_output_shapes
: R
	Y/Const_7Const*
_output_shapes
: *
dtype0*
valueB 2      №?R
	Y/Const_8Const*
_output_shapes
: *
dtype0*
valueB 2      №Пp
Y/SelectV2_2SelectV2Y/Equal_2:z:0Y/Const_7:output:0Y/Const_8:output:0*
T0*
_output_shapes
: R
	Y/Const_9Const*
_output_shapes
: *
dtype0*
valueB 2       @E
Y/Sqrt_6SqrtY/Const_9:output:0*
T0*
_output_shapes
: K
	Y/Abs_3/xConst*
_output_shapes
: *
dtype0*
value	B :C
Y/Abs_3AbsY/Abs_3/x:output:0*
T0*
_output_shapes
: K
	Y/mod_3/yConst*
_output_shapes
: *
dtype0*
value	B :U
Y/mod_3FloorModY/Abs_3:y:0Y/mod_3/y:output:0*
T0*
_output_shapes
: M
Y/Equal_3/yConst*
_output_shapes
: *
dtype0*
value	B : V
	Y/Equal_3EqualY/mod_3:z:0Y/Equal_3/y:output:0*
T0*
_output_shapes
: S

Y/Const_10Const*
_output_shapes
: *
dtype0*
valueB 2      №?S

Y/Const_11Const*
_output_shapes
: *
dtype0*
valueB 2      №Пr
Y/SelectV2_3SelectV2Y/Equal_3:z:0Y/Const_10:output:0Y/Const_11:output:0*
T0*
_output_shapes
: S

Y/Const_12Const*
_output_shapes
: *
dtype0*
valueB 2       @F
Y/Sqrt_7SqrtY/Const_12:output:0*
T0*
_output_shapes
: Y
Y/mul_25MulY/Sqrt_7:y:0Y/mul_13:z:0*
T0*#
_output_shapes
:џџџџџџџџџb
Y/mul_26MulY/mul_25:z:0Y/SelectV2_3:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
	Y/Abs_4/xConst*
_output_shapes
: *
dtype0*
valueB :
ўџџџџџџџџC
Y/Abs_4AbsY/Abs_4/x:output:0*
T0*
_output_shapes
: K
	Y/mod_4/yConst*
_output_shapes
: *
dtype0*
value	B :U
Y/mod_4FloorModY/Abs_4:y:0Y/mod_4/y:output:0*
T0*
_output_shapes
: M
Y/Equal_4/yConst*
_output_shapes
: *
dtype0*
value	B : V
	Y/Equal_4EqualY/mod_4:z:0Y/Equal_4/y:output:0*
T0*
_output_shapes
: S

Y/Const_13Const*
_output_shapes
: *
dtype0*
valueB 2      №?S

Y/Const_14Const*
_output_shapes
: *
dtype0*
valueB 2      №Пr
Y/SelectV2_4SelectV2Y/Equal_4:z:0Y/Const_13:output:0Y/Const_14:output:0*
T0*
_output_shapes
: S

Y/Const_15Const*
_output_shapes
: *
dtype0*
valueB 2       @F
Y/Sqrt_8SqrtY/Const_15:output:0*
T0*
_output_shapes
: Y
Y/mul_27MulY/Sqrt_8:y:0Y/mul_22:z:0*
T0*#
_output_shapes
:џџџџџџџџџb
Y/mul_28MulY/mul_27:z:0Y/SelectV2_4:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
	Y/Abs_5/xConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџC
Y/Abs_5AbsY/Abs_5/x:output:0*
T0*
_output_shapes
: K
	Y/mod_5/yConst*
_output_shapes
: *
dtype0*
value	B :U
Y/mod_5FloorModY/Abs_5:y:0Y/mod_5/y:output:0*
T0*
_output_shapes
: M
Y/Equal_5/yConst*
_output_shapes
: *
dtype0*
value	B : V
	Y/Equal_5EqualY/mod_5:z:0Y/Equal_5/y:output:0*
T0*
_output_shapes
: S

Y/Const_16Const*
_output_shapes
: *
dtype0*
valueB 2      №?S

Y/Const_17Const*
_output_shapes
: *
dtype0*
valueB 2      №Пr
Y/SelectV2_5SelectV2Y/Equal_5:z:0Y/Const_16:output:0Y/Const_17:output:0*
T0*
_output_shapes
: S

Y/Const_18Const*
_output_shapes
: *
dtype0*
valueB 2       @F
Y/Sqrt_9SqrtY/Const_18:output:0*
T0*
_output_shapes
: Y
Y/mul_29MulY/Sqrt_9:y:0Y/mul_16:z:0*
T0*#
_output_shapes
:џџџџџџџџџb
Y/mul_30MulY/mul_29:z:0Y/SelectV2_5:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
	Y/Abs_6/xConst*
_output_shapes
: *
dtype0*
value	B : C
Y/Abs_6AbsY/Abs_6/x:output:0*
T0*
_output_shapes
: K
	Y/mod_6/yConst*
_output_shapes
: *
dtype0*
value	B :U
Y/mod_6FloorModY/Abs_6:y:0Y/mod_6/y:output:0*
T0*
_output_shapes
: M
Y/Equal_6/yConst*
_output_shapes
: *
dtype0*
value	B : V
	Y/Equal_6EqualY/mod_6:z:0Y/Equal_6/y:output:0*
T0*
_output_shapes
: S

Y/Const_19Const*
_output_shapes
: *
dtype0*
valueB 2      №?S

Y/Const_20Const*
_output_shapes
: *
dtype0*
valueB 2      №Пr
Y/SelectV2_6SelectV2Y/Equal_6:z:0Y/Const_19:output:0Y/Const_20:output:0*
T0*
_output_shapes
: S

Y/Const_21Const*
_output_shapes
: *
dtype0*
valueB 2       @G
	Y/Sqrt_10SqrtY/Const_21:output:0*
T0*
_output_shapes
: K
	Y/Abs_7/xConst*
_output_shapes
: *
dtype0*
value	B :C
Y/Abs_7AbsY/Abs_7/x:output:0*
T0*
_output_shapes
: K
	Y/mod_7/yConst*
_output_shapes
: *
dtype0*
value	B :U
Y/mod_7FloorModY/Abs_7:y:0Y/mod_7/y:output:0*
T0*
_output_shapes
: M
Y/Equal_7/yConst*
_output_shapes
: *
dtype0*
value	B : V
	Y/Equal_7EqualY/mod_7:z:0Y/Equal_7/y:output:0*
T0*
_output_shapes
: S

Y/Const_22Const*
_output_shapes
: *
dtype0*
valueB 2      №?S

Y/Const_23Const*
_output_shapes
: *
dtype0*
valueB 2      №Пr
Y/SelectV2_7SelectV2Y/Equal_7:z:0Y/Const_22:output:0Y/Const_23:output:0*
T0*
_output_shapes
: S

Y/Const_24Const*
_output_shapes
: *
dtype0*
valueB 2       @G
	Y/Sqrt_11SqrtY/Const_24:output:0*
T0*
_output_shapes
: Z
Y/mul_31MulY/Sqrt_11:y:0Y/mul_15:z:0*
T0*#
_output_shapes
:џџџџџџџџџb
Y/mul_32MulY/mul_31:z:0Y/SelectV2_7:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
	Y/Abs_8/xConst*
_output_shapes
: *
dtype0*
value	B :C
Y/Abs_8AbsY/Abs_8/x:output:0*
T0*
_output_shapes
: K
	Y/mod_8/yConst*
_output_shapes
: *
dtype0*
value	B :U
Y/mod_8FloorModY/Abs_8:y:0Y/mod_8/y:output:0*
T0*
_output_shapes
: M
Y/Equal_8/yConst*
_output_shapes
: *
dtype0*
value	B : V
	Y/Equal_8EqualY/mod_8:z:0Y/Equal_8/y:output:0*
T0*
_output_shapes
: S

Y/Const_25Const*
_output_shapes
: *
dtype0*
valueB 2      №?S

Y/Const_26Const*
_output_shapes
: *
dtype0*
valueB 2      №Пr
Y/SelectV2_8SelectV2Y/Equal_8:z:0Y/Const_25:output:0Y/Const_26:output:0*
T0*
_output_shapes
: S

Y/Const_27Const*
_output_shapes
: *
dtype0*
valueB 2       @G
	Y/Sqrt_12SqrtY/Const_27:output:0*
T0*
_output_shapes
: Z
Y/mul_33MulY/Sqrt_12:y:0Y/mul_21:z:0*
T0*#
_output_shapes
:џџџџџџџџџb
Y/mul_34MulY/mul_33:z:0Y/SelectV2_8:output:0*
T0*#
_output_shapes
:џџџџџџџџџѕ
	Y/stack_1PackY/strided_slice_6:output:0Y/mul_24:z:0Y/strided_slice_8:output:0Y/mul_26:z:0Y/mul_28:z:0Y/mul_30:z:0Y/strided_slice_10:output:0Y/mul_32:z:0Y/mul_34:z:0*
N	*
T0*'
_output_shapes
:	џџџџџџџџџa
Y/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       y
Y/transpose	TransposeY/stack_1:output:0Y/transpose/perm:output:0*
T0*'
_output_shapes
:џџџџџџџџџ	^
Y/mul_35MulY/transpose:y:0
y_mul_35_y*
T0*'
_output_shapes
:џџџџџџџџџ	
A/einsum/EinsumEinsumR/GatherV2:output:0Y/mul_35:z:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ	*
equationjnl,jl->jnl
ChemIndTransf/ReadVariableOpReadVariableOp%chemindtransf_readvariableop_resource*
_output_shapes

:*
dtype0|
ChemIndTransf/mulMul$ChemIndTransf/ReadVariableOp:value:0chemindtransf_mul_y*
T0*
_output_shapes

:
*ChemIndTransf/einsum/Einsum/ReadVariableOpReadVariableOp3chemindtransf_einsum_einsum_readvariableop_resource*
_output_shapes

:*
dtype0Т
ChemIndTransf/einsum/EinsumEinsum2ChemIndTransf/einsum/Einsum/ReadVariableOp:value:0ChemIndTransf/mul:z:0*
N*
T0*
_output_shapes

:*
equation...k,...kn->...nQ
A/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Б

A/GatherV2GatherV2$ChemIndTransf/einsum/Einsum:output:0mu_jA/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:џџџџџџџџџЄ
A/einsum_1/EinsumEinsumA/einsum/Einsum:output:0A/GatherV2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ	*
equationjnl,jn->jnlЂ
A/UnsortedSegmentSumUnsortedSegmentSumA/einsum_1/Einsum:output:0ind_ibatch_tot_nat*
Tindices0*
T0*+
_output_shapes
:џџџџџџџџџ	j
A/mulMulA/UnsortedSegmentSum:output:0a_mul_y*
T0*+
_output_shapes
:џџџџџџџџџ	T
rho/zeros/packed/0Const*
_output_shapes
: *
dtype0*
value	B :T
rho/zeros/packed/2Const*
_output_shapes
: *
dtype0*
value	B :
rho/zeros/packedPackrho/zeros/packed/0:output:0batch_tot_natrho/zeros/packed/2:output:0*
N*
T0*
_output_shapes
:X
rho/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        |
	rho/zerosFillrho/zeros/packed:output:0rho/zeros/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
rho/GatherV2/indicesConst*
_output_shapes
:*
dtype0	*
valueB	R S
rho/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :З
rho/GatherV2GatherV2	A/mul:z:0rho/GatherV2/indices:output:0rho/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџ
rho/GatherV2_1/ReadVariableOpReadVariableOp&rho_gatherv2_1_readvariableop_resource*&
_output_shapes
:*
dtype0^
rho/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЫ
rho/GatherV2_1GatherV2%rho/GatherV2_1/ReadVariableOp:value:0rho_gatherv2_1_indicesrho/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*&
_output_shapes
:U
rho/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
rho/GatherV2_2GatherV2rho/GatherV2_1:output:0atomic_mu_irho/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:џџџџџџџџџІ
rho/ein_A/EinsumEinsumrho/GatherV2_2:output:0rho/GatherV2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationaknw,anw->wakj
rho/mulMulrho/ein_A/Einsum:output:0	rho_mul_y*
T0*+
_output_shapes
:џџџџџџџџџ­
rho/TensorScatterAddTensorScatterAddrho/zeros:output:0rho_tensorscatteradd_indicesrho/mul:z:0*
Tindices0*
T0*+
_output_shapes
:џџџџџџџџџr
	rho/mul_1Mulrho/TensorScatterAdd:output:0rho_mul_1_y*
T0*+
_output_shapes
:џџџџџџџџџg
rho/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
rho/transpose	Transposerho/mul_1:z:0rho/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
strided_sliceStridedSlicerho/transpose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ќ
strided_slice_1StridedSlicestrided_slice:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskN
add/xConst*
_output_shapes
: *
dtype0*
valueB 2        h
addAddV2add/x:output:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_2StridedSlicestrided_slice:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   v
ReshapeReshapestrided_slice_2:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџP
add_1/xConst*
_output_shapes
: *
dtype0*
valueB 2        d
add_1AddV2add_1/x:output:0Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
DenseLayer/ReadVariableOp_4ReadVariableOp$denselayer_readvariableop_4_resource*
_output_shapes

:@*
dtype0{
DenseLayer/mul_13Mul#DenseLayer/ReadVariableOp_4:value:0denselayer_mul_13_y*
T0*
_output_shapes

:@
DenseLayer/einsum_4/EinsumEinsumadd:z:0DenseLayer/mul_13:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@*
equation...k,...kn->...nV
DenseLayer/beta_3Const*
_output_shapes
: *
dtype0*
valueB
 *  ?e
DenseLayer/Cast_3CastDenseLayer/beta_3:output:0*

DstT0*

SrcT0*
_output_shapes
: 
DenseLayer/mul_14MulDenseLayer/Cast_3:y:0#DenseLayer/einsum_4/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@h
DenseLayer/Sigmoid_3SigmoidDenseLayer/mul_14:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
DenseLayer/mul_15Mul#DenseLayer/einsum_4/Einsum:output:0DenseLayer/Sigmoid_3:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@j
DenseLayer/Identity_3IdentityDenseLayer/mul_15:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@ю
DenseLayer/IdentityN_3	IdentityNDenseLayer/mul_15:z:0#DenseLayer/einsum_4/Einsum:output:0DenseLayer/Cast_3:y:0*
T
2**
_gradient_op_typeCustomGradient-8663*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \
DenseLayer/mul_16/yConst*
_output_shapes
: *
dtype0*
valueB 2ЦмЕ|ањ?
DenseLayer/mul_16MulDenseLayer/IdentityN_3:output:0DenseLayer/mul_16/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
DenseLayer/ReadVariableOp_5ReadVariableOp$denselayer_readvariableop_5_resource*
_output_shapes

:@*
dtype0{
DenseLayer/mul_17Mul#DenseLayer/ReadVariableOp_5:value:0denselayer_mul_17_y*
T0*
_output_shapes

:@­
DenseLayer/einsum_5/EinsumEinsumDenseLayer/mul_16:z:0DenseLayer/mul_17:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ*
equation...k,...kn->...np
add_2AddV2#DenseLayer/einsum_5/Einsum:output:0	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN
mulMul	add_2:z:0mul_y*
T0*'
_output_shapes
:џџџџџџџџџN
ConstConst*
_output_shapes
: *
dtype0*
valueB 2      №?W
mul_1Mulmul:z:0Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_3AddV2add_3_x	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџR
mul_2Mul	add_3:z:0mul_2_y*
T0*'
_output_shapes
:џџџџџџџџџ`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   k
	Reshape_1Reshape	mul_2:z:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ_
ones_like/ShapeShapeReshape_1:output:0*
T0*
_output_shapes
::эЯX
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB 2      №?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
gradient_tape/ShapeShape	mul_2:z:0*
T0*
_output_shapes
::эЯ
gradient_tape/ReshapeReshapeones_like:output:0gradient_tape/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџy
gradient_tape/mul_2/MulMulgradient_tape/Reshape:output:0mul_2_y*
T0*'
_output_shapes
:џџџџџџџџџ\
gradient_tape/add_3/ShapeConst*
_output_shapes
: *
dtype0*
valueB b
gradient_tape/add_3/Shape_1Shape	mul_1:z:0*
T0*
_output_shapes
::эЯ}
gradient_tape/mul_1/MulMulgradient_tape/mul_2/Mul:z:0Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџr
gradient_tape/mul/MulMulgradient_tape/mul_1/Mul:z:0mul_y*
T0*'
_output_shapes
:џџџџџџџџџz
gradient_tape/add_2/ShapeShape#DenseLayer/einsum_5/Einsum:output:0*
T0*
_output_shapes
::эЯb
gradient_tape/add_2/Shape_1Shape	add_1:z:0*
T0*
_output_shapes
::эЯР
)gradient_tape/add_2/BroadcastGradientArgsBroadcastGradientArgs"gradient_tape/add_2/Shape:output:0$gradient_tape/add_2/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЕ
gradient_tape/add_2/SumSumgradient_tape/mul/Mul:z:0.gradient_tape/add_2/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
gradient_tape/add_2/ReshapeReshape gradient_tape/add_2/Sum:output:0"gradient_tape/add_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЗ
gradient_tape/add_2/Sum_1Sumgradient_tape/mul/Mul:z:0.gradient_tape/add_2/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(Є
gradient_tape/add_2/Reshape_1Reshape"gradient_tape/add_2/Sum_1:output:0$gradient_tape/add_2/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџz
'gradient_tape/DenseLayer/einsum_5/ShapeShapeDenseLayer/mul_16:z:0*
T0*
_output_shapes
::эЯz
)gradient_tape/DenseLayer/einsum_5/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"@      Ъ
(gradient_tape/DenseLayer/einsum_5/EinsumEinsum$gradient_tape/add_2/Reshape:output:0DenseLayer/mul_17:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@*
equation...n,...kn->...kа
*gradient_tape/DenseLayer/einsum_5/Einsum_1Einsum$gradient_tape/add_2/Reshape:output:0DenseLayer/mul_16:z:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ@*
equation...n,...k->...kn
5gradient_tape/DenseLayer/einsum_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7gradient_tape/DenseLayer/einsum_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
7gradient_tape/DenseLayer/einsum_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
/gradient_tape/DenseLayer/einsum_5/strided_sliceStridedSlice0gradient_tape/DenseLayer/einsum_5/Shape:output:0>gradient_tape/DenseLayer/einsum_5/strided_slice/stack:output:0@gradient_tape/DenseLayer/einsum_5/strided_slice/stack_1:output:0@gradient_tape/DenseLayer/einsum_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:
7gradient_tape/DenseLayer/einsum_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9gradient_tape/DenseLayer/einsum_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ
9gradient_tape/DenseLayer/einsum_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
1gradient_tape/DenseLayer/einsum_5/strided_slice_1StridedSlice2gradient_tape/DenseLayer/einsum_5/Shape_1:output:0@gradient_tape/DenseLayer/einsum_5/strided_slice_1/stack:output:0Bgradient_tape/DenseLayer/einsum_5/strided_slice_1/stack_1:output:0Bgradient_tape/DenseLayer/einsum_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: њ
7gradient_tape/DenseLayer/einsum_5/BroadcastGradientArgsBroadcastGradientArgs8gradient_tape/DenseLayer/einsum_5/strided_slice:output:0:gradient_tape/DenseLayer/einsum_5/strided_slice_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџi
'gradient_tape/DenseLayer/einsum_5/add/xConst*
_output_shapes
: *
dtype0*
value	B : Ь
%gradient_tape/DenseLayer/einsum_5/addAddV20gradient_tape/DenseLayer/einsum_5/add/x:output:0<gradient_tape/DenseLayer/einsum_5/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ­
%gradient_tape/DenseLayer/einsum_5/SumSum1gradient_tape/DenseLayer/einsum_5/Einsum:output:0)gradient_tape/DenseLayer/einsum_5/add:z:0*
T0*
_output_shapes
:Ш
)gradient_tape/DenseLayer/einsum_5/ReshapeReshape.gradient_tape/DenseLayer/einsum_5/Sum:output:00gradient_tape/DenseLayer/einsum_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@k
)gradient_tape/DenseLayer/einsum_5/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : а
'gradient_tape/DenseLayer/einsum_5/add_1AddV22gradient_tape/DenseLayer/einsum_5/add_1/x:output:0<gradient_tape/DenseLayer/einsum_5/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџГ
'gradient_tape/DenseLayer/einsum_5/Sum_1Sum3gradient_tape/DenseLayer/einsum_5/Einsum_1:output:0+gradient_tape/DenseLayer/einsum_5/add_1:z:0*
T0*
_output_shapes
:Х
+gradient_tape/DenseLayer/einsum_5/Reshape_1Reshape0gradient_tape/DenseLayer/einsum_5/Sum_1:output:02gradient_tape/DenseLayer/einsum_5/Shape_1:output:0*
T0*
_output_shapes

:@\
gradient_tape/add_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB i
gradient_tape/add_1/Shape_1ShapeReshape:output:0*
T0*
_output_shapes
::эЯж
#gradient_tape/DenseLayer/mul_16/MulMul2gradient_tape/DenseLayer/einsum_5/Reshape:output:0DenseLayer/mul_16/y:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ@
#gradient_tape/DenseLayer/mul_17/MulMul4gradient_tape/DenseLayer/einsum_5/Reshape_1:output:0denselayer_mul_17_y*
T0*
_output_shapes

:@k
gradient_tape/Shape_1Shapestrided_slice_2:output:0*
T0*
_output_shapes
::эЯ
gradient_tape/Reshape_1Reshape&gradient_tape/add_2/Reshape_1:output:0gradient_tape/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџj

zeros_like	ZerosLikeDenseLayer/IdentityN_3:output:1*
T0*'
_output_shapes
:џџџџџџџџџ@N
zerosConst*
_output_shapes
: *
dtype0*
valueB 2         
mul_3MulDenseLayer/Cast_3:y:0#DenseLayer/einsum_4/Einsum:output:0$^gradient_tape/DenseLayer/mul_16/Mul*
T0*'
_output_shapes
:џџџџџџџџџ@O
SigmoidSigmoid	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@z
mul_4MulDenseLayer/Cast_3:y:0#DenseLayer/einsum_4/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
sub/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_5Mul	mul_4:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
add_4/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?]
add_4AddV2add_4/x:output:0	mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_6MulSigmoid:y:0	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@g
SquareSquare#DenseLayer/einsum_4/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@s
mul_7Mul'gradient_tape/DenseLayer/mul_16/Mul:z:0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_8Mul	mul_7:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?]
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_9Mul	mul_8:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       H
SumSum	mul_9:z:0Const_1:output:0*
T0*
_output_shapes
: s
mul_10Mul'gradient_tape/DenseLayer/mul_16/Mul:z:0	mul_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
#gradient_tape/strided_slice_2/ShapeShapestrided_slice:output:0*
T0*
_output_shapes
::эЯ
4gradient_tape/strided_slice_2/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB"        
2gradient_tape/strided_slice_2/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB"       
6gradient_tape/strided_slice_2/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB"      Ш
.gradient_tape/strided_slice_2/StridedSliceGradStridedSliceGrad,gradient_tape/strided_slice_2/Shape:output:0=gradient_tape/strided_slice_2/StridedSliceGrad/begin:output:0;gradient_tape/strided_slice_2/StridedSliceGrad/end:output:0?gradient_tape/strided_slice_2/StridedSliceGrad/strides:output:0 gradient_tape/Reshape_1:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskl
'gradient_tape/DenseLayer/einsum_4/ShapeShapeadd:z:0*
T0*
_output_shapes
::эЯz
)gradient_tape/DenseLayer/einsum_4/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   @   А
(gradient_tape/DenseLayer/einsum_4/EinsumEinsum
mul_10:z:0DenseLayer/mul_13:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ*
equation...n,...kn->...kЈ
*gradient_tape/DenseLayer/einsum_4/Einsum_1Einsum
mul_10:z:0add:z:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ@*
equation...n,...k->...kn
5gradient_tape/DenseLayer/einsum_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7gradient_tape/DenseLayer/einsum_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
7gradient_tape/DenseLayer/einsum_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
/gradient_tape/DenseLayer/einsum_4/strided_sliceStridedSlice0gradient_tape/DenseLayer/einsum_4/Shape:output:0>gradient_tape/DenseLayer/einsum_4/strided_slice/stack:output:0@gradient_tape/DenseLayer/einsum_4/strided_slice/stack_1:output:0@gradient_tape/DenseLayer/einsum_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:
7gradient_tape/DenseLayer/einsum_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9gradient_tape/DenseLayer/einsum_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ
9gradient_tape/DenseLayer/einsum_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
1gradient_tape/DenseLayer/einsum_4/strided_slice_1StridedSlice2gradient_tape/DenseLayer/einsum_4/Shape_1:output:0@gradient_tape/DenseLayer/einsum_4/strided_slice_1/stack:output:0Bgradient_tape/DenseLayer/einsum_4/strided_slice_1/stack_1:output:0Bgradient_tape/DenseLayer/einsum_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: њ
7gradient_tape/DenseLayer/einsum_4/BroadcastGradientArgsBroadcastGradientArgs8gradient_tape/DenseLayer/einsum_4/strided_slice:output:0:gradient_tape/DenseLayer/einsum_4/strided_slice_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџi
'gradient_tape/DenseLayer/einsum_4/add/xConst*
_output_shapes
: *
dtype0*
value	B : Ь
%gradient_tape/DenseLayer/einsum_4/addAddV20gradient_tape/DenseLayer/einsum_4/add/x:output:0<gradient_tape/DenseLayer/einsum_4/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ­
%gradient_tape/DenseLayer/einsum_4/SumSum1gradient_tape/DenseLayer/einsum_4/Einsum:output:0)gradient_tape/DenseLayer/einsum_4/add:z:0*
T0*
_output_shapes
:Ш
)gradient_tape/DenseLayer/einsum_4/ReshapeReshape.gradient_tape/DenseLayer/einsum_4/Sum:output:00gradient_tape/DenseLayer/einsum_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџk
)gradient_tape/DenseLayer/einsum_4/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : а
'gradient_tape/DenseLayer/einsum_4/add_1AddV22gradient_tape/DenseLayer/einsum_4/add_1/x:output:0<gradient_tape/DenseLayer/einsum_4/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџГ
'gradient_tape/DenseLayer/einsum_4/Sum_1Sum3gradient_tape/DenseLayer/einsum_4/Einsum_1:output:0+gradient_tape/DenseLayer/einsum_4/add_1:z:0*
T0*
_output_shapes
:Х
+gradient_tape/DenseLayer/einsum_4/Reshape_1Reshape0gradient_tape/DenseLayer/einsum_4/Sum_1:output:02gradient_tape/DenseLayer/einsum_4/Shape_1:output:0*
T0*
_output_shapes

:@Z
gradient_tape/add/ShapeConst*
_output_shapes
: *
dtype0*
valueB o
gradient_tape/add/Shape_1Shapestrided_slice_1:output:0*
T0*
_output_shapes
::эЯ
#gradient_tape/DenseLayer/mul_13/MulMul4gradient_tape/DenseLayer/einsum_4/Reshape_1:output:0denselayer_mul_13_y*
T0*
_output_shapes

:@w
#gradient_tape/strided_slice_1/ShapeShapestrided_slice:output:0*
T0*
_output_shapes
::эЯ
4gradient_tape/strided_slice_1/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB"       
2gradient_tape/strided_slice_1/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB"        
6gradient_tape/strided_slice_1/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB"      Т
.gradient_tape/strided_slice_1/StridedSliceGradStridedSliceGrad,gradient_tape/strided_slice_1/Shape:output:0=gradient_tape/strided_slice_1/StridedSliceGrad/begin:output:0;gradient_tape/strided_slice_1/StridedSliceGrad/end:output:0?gradient_tape/strided_slice_1/StridedSliceGrad/strides:output:02gradient_tape/DenseLayer/einsum_4/Reshape:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskЙ
AddNAddN7gradient_tape/strided_slice_2/StridedSliceGrad:output:07gradient_tape/strided_slice_1/StridedSliceGrad:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџp
!gradient_tape/strided_slice/ShapeShaperho/transpose:y:0*
T0*
_output_shapes
::эЯ
2gradient_tape/strided_slice/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*!
valueB"            
0gradient_tape/strided_slice/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*!
valueB"           
4gradient_tape/strided_slice/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*!
valueB"         Ќ
,gradient_tape/strided_slice/StridedSliceGradStridedSliceGrad*gradient_tape/strided_slice/Shape:output:0;gradient_tape/strided_slice/StridedSliceGrad/begin:output:09gradient_tape/strided_slice/StridedSliceGrad/end:output:0=gradient_tape/strided_slice/StridedSliceGrad/strides:output:0
AddN:sum:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask{
-gradient_tape/rho/transpose/InvertPermutationInvertPermutationrho/transpose/perm:output:0*
_output_shapes
:в
%gradient_tape/rho/transpose/transpose	Transpose5gradient_tape/strided_slice/StridedSliceGrad:output:01gradient_tape/rho/transpose/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/rho/mul_1/MulMul)gradient_tape/rho/transpose/transpose:y:0rho_mul_1_y*
T0*+
_output_shapes
:џџџџџџџџџБ
gradient_tape/rho/GatherNdGatherNdgradient_tape/rho/mul_1/Mul:z:0rho_tensorscatteradd_indices*
Tindices0*
Tparams0*+
_output_shapes
:џџџџџџџџџ}
gradient_tape/rho/IdentityIdentitygradient_tape/rho/mul_1/Mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/rho/mul/MulMul#gradient_tape/rho/GatherNd:output:0	rho_mul_y*
T0*+
_output_shapes
:џџџџџџџџџr
gradient_tape/rho/ein_A/ShapeShaperho/GatherV2_2:output:0*
T0*
_output_shapes
::эЯr
gradient_tape/rho/ein_A/Shape_1Shaperho/GatherV2:output:0*
T0*
_output_shapes
::эЯО
gradient_tape/rho/ein_A/EinsumEinsumgradient_tape/rho/mul/Mul:z:0rho/GatherV2:output:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ*
equationwak,anw->aknwО
 gradient_tape/rho/ein_A/Einsum_1Einsumgradient_tape/rho/mul/Mul:z:0rho/GatherV2_2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationwak,aknw->anwЃ
gradient_tape/rho/ShapeConst*!
_class
loc:@rho/GatherV2_1*
_output_shapes
:*
dtype0	*5
value,B*	"                             
gradient_tape/rho/CastCast gradient_tape/rho/Shape:output:0*

DstT0*

SrcT0	*!
_class
loc:@rho/GatherV2_1*
_output_shapes
:L
gradient_tape/rho/SizeSizeatomic_mu_i*
T0*
_output_shapes
: b
 gradient_tape/rho/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho/ExpandDims
ExpandDimsgradient_tape/rho/Size:output:0)gradient_tape/rho/ExpandDims/dim:output:0*
T0*
_output_shapes
:o
%gradient_tape/rho/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:q
'gradient_tape/rho/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'gradient_tape/rho/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
gradient_tape/rho/strided_sliceStridedSlicegradient_tape/rho/Cast:y:0.gradient_tape/rho/strided_slice/stack:output:00gradient_tape/rho/strided_slice/stack_1:output:00gradient_tape/rho/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask_
gradient_tape/rho/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы
gradient_tape/rho/concatConcatV2%gradient_tape/rho/ExpandDims:output:0(gradient_tape/rho/strided_slice:output:0&gradient_tape/rho/concat/axis:output:0*
N*
T0*
_output_shapes
:Њ
gradient_tape/rho/ReshapeReshape'gradient_tape/rho/ein_A/Einsum:output:0!gradient_tape/rho/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
gradient_tape/rho/Reshape_1Reshapeatomic_mu_i%gradient_tape/rho/ExpandDims:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/rho/Shape_1Shape	A/mul:z:0*
T0*
_class

loc:@A/mul*
_output_shapes
:*
out_type0	:эа
gradient_tape/rho/Cast_1Cast"gradient_tape/rho/Shape_1:output:0*

DstT0*

SrcT0	*
_class

loc:@A/mul*
_output_shapes
:Z
gradient_tape/rho/Size_1Const*
_output_shapes
: *
dtype0*
value	B :d
"gradient_tape/rho/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ё
gradient_tape/rho/ExpandDims_1
ExpandDims!gradient_tape/rho/Size_1:output:0+gradient_tape/rho/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:Y
gradient_tape/rho/ConstConst*
_output_shapes
: *
dtype0*
value	B : [
gradient_tape/rho/Const_1Const*
_output_shapes
: *
dtype0*
value	B :
'gradient_tape/rho/strided_slice_1/stackPack gradient_tape/rho/Const:output:0*
N*
T0*
_output_shapes
:{
)gradient_tape/rho/strided_slice_1/stack_1Packrho/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:
)gradient_tape/rho/strided_slice_1/stack_2Pack"gradient_tape/rho/Const_1:output:0*
N*
T0*
_output_shapes
:­
!gradient_tape/rho/strided_slice_1StridedSlicegradient_tape/rho/Cast_1:y:00gradient_tape/rho/strided_slice_1/stack:output:02gradient_tape/rho/strided_slice_1/stack_1:output:02gradient_tape/rho/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask[
gradient_tape/rho/Const_2Const*
_output_shapes
: *
dtype0*
value	B : [
gradient_tape/rho/Const_3Const*
_output_shapes
: *
dtype0*
value	B :y
'gradient_tape/rho/strided_slice_2/stackPackrho/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:
)gradient_tape/rho/strided_slice_2/stack_1Pack"gradient_tape/rho/Const_2:output:0*
N*
T0*
_output_shapes
:
)gradient_tape/rho/strided_slice_2/stack_2Pack"gradient_tape/rho/Const_3:output:0*
N*
T0*
_output_shapes
:Ћ
!gradient_tape/rho/strided_slice_2StridedSlicegradient_tape/rho/Cast_1:y:00gradient_tape/rho/strided_slice_2/stack:output:02gradient_tape/rho/strided_slice_2/stack_1:output:02gradient_tape/rho/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskq
'gradient_tape/rho/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:s
)gradient_tape/rho/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)gradient_tape/rho/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
!gradient_tape/rho/strided_slice_3StridedSlice*gradient_tape/rho/strided_slice_2:output:00gradient_tape/rho/strided_slice_3/stack:output:02gradient_tape/rho/strided_slice_3/stack_1:output:02gradient_tape/rho/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskv
#gradient_tape/rho/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
gradient_tape/rho/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho/concat_1ConcatV2*gradient_tape/rho/strided_slice_1:output:0,gradient_tape/rho/concat_1/values_1:output:0*gradient_tape/rho/strided_slice_3:output:0(gradient_tape/rho/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Z
gradient_tape/rho/Size_2Const*
_output_shapes
: *
dtype0*
value	B :Z
gradient_tape/rho/Size_3Const*
_output_shapes
: *
dtype0*
value	B :_
gradient_tape/rho/range/startConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/rho/range/limitConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/rho/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :В
gradient_tape/rho/rangeRange&gradient_tape/rho/range/start:output:0&gradient_tape/rho/range/limit:output:0&gradient_tape/rho/range/delta:output:0*
_output_shapes
: a
gradient_tape/rho/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : a
gradient_tape/rho/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :Е
gradient_tape/rho/range_1Range(gradient_tape/rho/range_1/start:output:0!gradient_tape/rho/Size_3:output:0(gradient_tape/rho/range_1/delta:output:0*
_output_shapes
:Y
gradient_tape/rho/add/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/rho/addAddV2!gradient_tape/rho/Size_3:output:0 gradient_tape/rho/add/y:output:0*
T0*
_output_shapes
: a
gradient_tape/rho/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :Є
gradient_tape/rho/range_2Rangegradient_tape/rho/add:z:0!gradient_tape/rho/Size_2:output:0(gradient_tape/rho/range_2/delta:output:0*
_output_shapes
: Е
gradient_tape/rho/Reshape_2Reshape)gradient_tape/rho/ein_A/Einsum_1:output:0#gradient_tape/rho/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ|
#gradient_tape/rho/concat_2/values_1Pack!gradient_tape/rho/Size_3:output:0*
N*
T0*
_output_shapes
:a
gradient_tape/rho/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho/concat_2ConcatV2 gradient_tape/rho/range:output:0,gradient_tape/rho/concat_2/values_1:output:0"gradient_tape/rho/range_1:output:0"gradient_tape/rho/range_2:output:0(gradient_tape/rho/concat_2/axis:output:0*
N*
T0*
_output_shapes
:В
gradient_tape/rho/transpose	Transpose$gradient_tape/rho/Reshape_2:output:0#gradient_tape/rho/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџa
gradient_tape/rho/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : л
gradient_tape/rho/GatherV2GatherV2gradient_tape/rho/Cast_1:y:0#gradient_tape/rho/concat_2:output:0(gradient_tape/rho/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
gradient_tape/rho/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/rho/add_1AddV2rho/GatherV2/axis:output:0"gradient_tape/rho/add_1/y:output:0*
T0*
_output_shapes
: y
'gradient_tape/rho/strided_slice_4/stackPackrho/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:|
)gradient_tape/rho/strided_slice_4/stack_1Packgradient_tape/rho/add_1:z:0*
N*
T0*
_output_shapes
:s
)gradient_tape/rho/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Џ
!gradient_tape/rho/strided_slice_4StridedSlicegradient_tape/rho/Cast_1:y:00gradient_tape/rho/strided_slice_4/stack:output:02gradient_tape/rho/strided_slice_4/stack_1:output:02gradient_tape/rho/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
gradient_tape/rho/Size_4Const*
_output_shapes
: *
dtype0*
value	B :d
"gradient_tape/rho/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : Ё
gradient_tape/rho/ExpandDims_2
ExpandDims!gradient_tape/rho/Size_4:output:0+gradient_tape/rho/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:
gradient_tape/rho/Reshape_3Reshaperho/GatherV2/indices:output:0'gradient_tape/rho/ExpandDims_2:output:0*
T0	*
_output_shapes
:ќ
$gradient_tape/rho/UnsortedSegmentSumUnsortedSegmentSumgradient_tape/rho/transpose:y:0$gradient_tape/rho/Reshape_3:output:0*gradient_tape/rho/strided_slice_4:output:0*
Tindices0	*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ[
gradient_tape/rho/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/rho/add_2AddV2"gradient_tape/rho/range_1:output:0"gradient_tape/rho/add_2/y:output:0*
T0*
_output_shapes
:m
#gradient_tape/rho/concat_3/values_2Const*
_output_shapes
:*
dtype0*
valueB: a
gradient_tape/rho/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho/concat_3ConcatV2 gradient_tape/rho/range:output:0gradient_tape/rho/add_2:z:0,gradient_tape/rho/concat_3/values_2:output:0"gradient_tape/rho/range_2:output:0(gradient_tape/rho/concat_3/axis:output:0*
N*
T0*
_output_shapes
:Н
gradient_tape/rho/transpose_1	Transpose-gradient_tape/rho/UnsortedSegmentSum:output:0#gradient_tape/rho/concat_3:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџД
gradient_tape/rho/Shape_2Const*0
_class&
$"loc:@rho/GatherV2_1/ReadVariableOp*
_output_shapes
:*
dtype0	*5
value,B*	"                             Њ
gradient_tape/rho/Cast_2Cast"gradient_tape/rho/Shape_2:output:0*

DstT0*

SrcT0	*0
_class&
$"loc:@rho/GatherV2_1/ReadVariableOp*
_output_shapes
:Z
gradient_tape/rho/Size_5Const*
_output_shapes
: *
dtype0*
value	B :d
"gradient_tape/rho/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : Ё
gradient_tape/rho/ExpandDims_3
ExpandDims!gradient_tape/rho/Size_5:output:0+gradient_tape/rho/ExpandDims_3/dim:output:0*
T0*
_output_shapes
:[
gradient_tape/rho/Const_4Const*
_output_shapes
: *
dtype0*
value	B : [
gradient_tape/rho/Const_5Const*
_output_shapes
: *
dtype0*
value	B :
'gradient_tape/rho/strided_slice_5/stackPack"gradient_tape/rho/Const_4:output:0*
N*
T0*
_output_shapes
:}
)gradient_tape/rho/strided_slice_5/stack_1Packrho/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:
)gradient_tape/rho/strided_slice_5/stack_2Pack"gradient_tape/rho/Const_5:output:0*
N*
T0*
_output_shapes
:­
!gradient_tape/rho/strided_slice_5StridedSlicegradient_tape/rho/Cast_2:y:00gradient_tape/rho/strided_slice_5/stack:output:02gradient_tape/rho/strided_slice_5/stack_1:output:02gradient_tape/rho/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask[
gradient_tape/rho/Const_6Const*
_output_shapes
: *
dtype0*
value	B : [
gradient_tape/rho/Const_7Const*
_output_shapes
: *
dtype0*
value	B :{
'gradient_tape/rho/strided_slice_6/stackPackrho/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:
)gradient_tape/rho/strided_slice_6/stack_1Pack"gradient_tape/rho/Const_6:output:0*
N*
T0*
_output_shapes
:
)gradient_tape/rho/strided_slice_6/stack_2Pack"gradient_tape/rho/Const_7:output:0*
N*
T0*
_output_shapes
:Ћ
!gradient_tape/rho/strided_slice_6StridedSlicegradient_tape/rho/Cast_2:y:00gradient_tape/rho/strided_slice_6/stack:output:02gradient_tape/rho/strided_slice_6/stack_1:output:02gradient_tape/rho/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskq
'gradient_tape/rho/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:s
)gradient_tape/rho/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)gradient_tape/rho/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
!gradient_tape/rho/strided_slice_7StridedSlice*gradient_tape/rho/strided_slice_6:output:00gradient_tape/rho/strided_slice_7/stack:output:02gradient_tape/rho/strided_slice_7/stack_1:output:02gradient_tape/rho/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskv
#gradient_tape/rho/concat_4/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
gradient_tape/rho/concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho/concat_4ConcatV2*gradient_tape/rho/strided_slice_5:output:0,gradient_tape/rho/concat_4/values_1:output:0*gradient_tape/rho/strided_slice_7:output:0(gradient_tape/rho/concat_4/axis:output:0*
N*
T0*
_output_shapes
:Z
gradient_tape/rho/Size_6Const*
_output_shapes
: *
dtype0*
value	B :Z
gradient_tape/rho/Size_7Const*
_output_shapes
: *
dtype0*
value	B :a
gradient_tape/rho/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : a
gradient_tape/rho/range_3/limitConst*
_output_shapes
: *
dtype0*
value	B : a
gradient_tape/rho/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :К
gradient_tape/rho/range_3Range(gradient_tape/rho/range_3/start:output:0(gradient_tape/rho/range_3/limit:output:0(gradient_tape/rho/range_3/delta:output:0*
_output_shapes
: a
gradient_tape/rho/range_4/startConst*
_output_shapes
: *
dtype0*
value	B : a
gradient_tape/rho/range_4/deltaConst*
_output_shapes
: *
dtype0*
value	B :Е
gradient_tape/rho/range_4Range(gradient_tape/rho/range_4/start:output:0!gradient_tape/rho/Size_7:output:0(gradient_tape/rho/range_4/delta:output:0*
_output_shapes
:[
gradient_tape/rho/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/rho/add_3AddV2!gradient_tape/rho/Size_7:output:0"gradient_tape/rho/add_3/y:output:0*
T0*
_output_shapes
: a
gradient_tape/rho/range_5/deltaConst*
_output_shapes
: *
dtype0*
value	B :І
gradient_tape/rho/range_5Rangegradient_tape/rho/add_3:z:0!gradient_tape/rho/Size_6:output:0(gradient_tape/rho/range_5/delta:output:0*
_output_shapes
: q
'gradient_tape/rho/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)gradient_tape/rho/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)gradient_tape/rho/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
!gradient_tape/rho/strided_slice_8StridedSlicegradient_tape/rho/Cast:y:00gradient_tape/rho/strided_slice_8/stack:output:02gradient_tape/rho/strided_slice_8/stack_1:output:02gradient_tape/rho/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskѓ
&gradient_tape/rho/UnsortedSegmentSum_1UnsortedSegmentSum"gradient_tape/rho/Reshape:output:0$gradient_tape/rho/Reshape_1:output:0*gradient_tape/rho/strided_slice_8:output:0*
Tindices0*
T0*&
_output_shapes
:­
gradient_tape/rho/Reshape_4Reshape/gradient_tape/rho/UnsortedSegmentSum_1:output:0#gradient_tape/rho/concat_4:output:0*
T0*&
_output_shapes
:|
#gradient_tape/rho/concat_5/values_1Pack!gradient_tape/rho/Size_7:output:0*
N*
T0*
_output_shapes
:a
gradient_tape/rho/concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho/concat_5ConcatV2"gradient_tape/rho/range_3:output:0,gradient_tape/rho/concat_5/values_1:output:0"gradient_tape/rho/range_4:output:0"gradient_tape/rho/range_5:output:0(gradient_tape/rho/concat_5/axis:output:0*
N*
T0*
_output_shapes
:І
gradient_tape/rho/transpose_2	Transpose$gradient_tape/rho/Reshape_4:output:0#gradient_tape/rho/concat_5:output:0*
T0*&
_output_shapes
:c
!gradient_tape/rho/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : п
gradient_tape/rho/GatherV2_1GatherV2gradient_tape/rho/Cast_2:y:0#gradient_tape/rho/concat_5:output:0*gradient_tape/rho/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
gradient_tape/rho/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/rho/add_4AddV2rho/GatherV2_1/axis:output:0"gradient_tape/rho/add_4/y:output:0*
T0*
_output_shapes
: {
'gradient_tape/rho/strided_slice_9/stackPackrho/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:|
)gradient_tape/rho/strided_slice_9/stack_1Packgradient_tape/rho/add_4:z:0*
N*
T0*
_output_shapes
:s
)gradient_tape/rho/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Џ
!gradient_tape/rho/strided_slice_9StridedSlicegradient_tape/rho/Cast_2:y:00gradient_tape/rho/strided_slice_9/stack:output:02gradient_tape/rho/strided_slice_9/stack_1:output:02gradient_tape/rho/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
gradient_tape/rho/Size_8Const*
_output_shapes
: *
dtype0*
value	B :d
"gradient_tape/rho/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B : Ё
gradient_tape/rho/ExpandDims_4
ExpandDims!gradient_tape/rho/Size_8:output:0+gradient_tape/rho/ExpandDims_4/dim:output:0*
T0*
_output_shapes
:
gradient_tape/rho/Reshape_5Reshaperho_gatherv2_1_indices'gradient_tape/rho/ExpandDims_4:output:0*
T0*
_output_shapes
:ђ
&gradient_tape/rho/UnsortedSegmentSum_2UnsortedSegmentSum!gradient_tape/rho/transpose_2:y:0$gradient_tape/rho/Reshape_5:output:0*gradient_tape/rho/strided_slice_9:output:0*
Tindices0*
T0*&
_output_shapes
:[
gradient_tape/rho/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/rho/add_5AddV2"gradient_tape/rho/range_4:output:0"gradient_tape/rho/add_5/y:output:0*
T0*
_output_shapes
:m
#gradient_tape/rho/concat_6/values_2Const*
_output_shapes
:*
dtype0*
valueB: a
gradient_tape/rho/concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho/concat_6ConcatV2"gradient_tape/rho/range_3:output:0gradient_tape/rho/add_5:z:0,gradient_tape/rho/concat_6/values_2:output:0"gradient_tape/rho/range_5:output:0(gradient_tape/rho/concat_6/axis:output:0*
N*
T0*
_output_shapes
:Б
gradient_tape/rho/transpose_3	Transpose/gradient_tape/rho/UnsortedSegmentSum_2:output:0#gradient_tape/rho/concat_6:output:0*
T0*&
_output_shapes
:
gradient_tape/A/mul/MulMul!gradient_tape/rho/transpose_1:y:0a_mul_y*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ\
gradient_tape/A/zeros_like	ZerosLikeind_i*
T0*#
_output_shapes
:џџџџџџџџџw
gradient_tape/A/MaximumMaximumind_igradient_tape/A/zeros_like:y:0*
T0*#
_output_shapes
:џџџџџџџџџ_
gradient_tape/A/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
gradient_tape/A/GatherV2GatherV2gradient_tape/A/mul/Mul:z:0gradient_tape/A/Maximum:z:0&gradient_tape/A/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ`
gradient_tape/A/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/A/GreaterEqualGreaterEqualind_i'gradient_tape/A/GreaterEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџs
gradient_tape/A/ShapeShape gradient_tape/A/GreaterEqual:z:0*
T0
*
_output_shapes
::эЯV
gradient_tape/A/RankConst*
_output_shapes
: *
dtype0*
value	B :X
gradient_tape/A/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :{
gradient_tape/A/subSubgradient_tape/A/Rank:output:0gradient_tape/A/Rank_1:output:0*
T0*
_output_shapes
: j
gradient_tape/A/ones/packedPackgradient_tape/A/sub:z:0*
N*
T0*
_output_shapes
:\
gradient_tape/A/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/A/onesFill$gradient_tape/A/ones/packed:output:0#gradient_tape/A/ones/Const:output:0*
T0*
_output_shapes
:]
gradient_tape/A/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Е
gradient_tape/A/concatConcatV2gradient_tape/A/Shape:output:0gradient_tape/A/ones:output:0$gradient_tape/A/concat/axis:output:0*
N*
T0*
_output_shapes
:
gradient_tape/A/ReshapeReshape gradient_tape/A/GreaterEqual:z:0gradient_tape/A/concat:output:0*
T0
*+
_output_shapes
:џџџџџџџџџ~
gradient_tape/A/ones_like/ShapeShape!gradient_tape/A/GatherV2:output:0*
T0*
_output_shapes
::эЯa
gradient_tape/A/ones_like/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 ZД
gradient_tape/A/ones_likeFill(gradient_tape/A/ones_like/Shape:output:0(gradient_tape/A/ones_like/Const:output:0*
T0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
gradient_tape/A/and
LogicalAnd gradient_tape/A/Reshape:output:0"gradient_tape/A/ones_like:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
gradient_tape/A/zeros_like_1	ZerosLike!gradient_tape/A/GatherV2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџС
gradient_tape/A/SelectV2SelectV2gradient_tape/A/and:z:0!gradient_tape/A/GatherV2:output:0 gradient_tape/A/zeros_like_1:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
gradient_tape/A/einsum_1/ShapeShapeA/einsum/Einsum:output:0*
T0*
_output_shapes
::эЯq
 gradient_tape/A/einsum_1/Shape_1ShapeA/GatherV2:output:0*
T0*
_output_shapes
::эЯФ
gradient_tape/A/einsum_1/EinsumEinsum!gradient_tape/A/SelectV2:output:0A/GatherV2:output:0*
N*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
equationjnl,jn->jnlО
!gradient_tape/A/einsum_1/Einsum_1Einsum!gradient_tape/A/SelectV2:output:0A/einsum/Einsum:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ*
equationjnl,jnl->jnm
gradient_tape/A/einsum/ShapeShapeR/GatherV2:output:0*
T0*
_output_shapes
::эЯh
gradient_tape/A/einsum/Shape_1ShapeY/mul_35:z:0*
T0*
_output_shapes
::эЯЙ
gradient_tape/A/einsum/EinsumEinsum(gradient_tape/A/einsum_1/Einsum:output:0Y/mul_35:z:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ	*
equationjnl,jl->jnlО
gradient_tape/A/einsum/Einsum_1Einsum(gradient_tape/A/einsum_1/Einsum:output:0R/GatherV2:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ	*
equationjnl,jnl->jl 
gradient_tape/A/Shape_1Const*.
_class$
" loc:@ChemIndTransf/einsum/Einsum*
_output_shapes
:*
dtype0	*%
valueB	"              Ђ
gradient_tape/A/CastCast gradient_tape/A/Shape_1:output:0*

DstT0*

SrcT0	*.
_class$
" loc:@ChemIndTransf/einsum/Einsum*
_output_shapes
:C
gradient_tape/A/SizeSizemu_j*
T0*
_output_shapes
: `
gradient_tape/A/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/A/ExpandDims
ExpandDimsgradient_tape/A/Size:output:0'gradient_tape/A/ExpandDims/dim:output:0*
T0*
_output_shapes
:m
#gradient_tape/A/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:o
%gradient_tape/A/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%gradient_tape/A/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gradient_tape/A/strided_sliceStridedSlicegradient_tape/A/Cast:y:0,gradient_tape/A/strided_slice/stack:output:0.gradient_tape/A/strided_slice/stack_1:output:0.gradient_tape/A/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask_
gradient_tape/A/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
gradient_tape/A/concat_1ConcatV2#gradient_tape/A/ExpandDims:output:0&gradient_tape/A/strided_slice:output:0&gradient_tape/A/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѕ
gradient_tape/A/Reshape_1Reshape*gradient_tape/A/einsum_1/Einsum_1:output:0!gradient_tape/A/concat_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ}
gradient_tape/A/Reshape_2Reshapemu_j#gradient_tape/A/ExpandDims:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/R/ShapeShapeR/Reshape:output:0*
T0*
_class
loc:@R/Reshape*
_output_shapes
:*
out_type0	:эа
gradient_tape/R/CastCastgradient_tape/R/Shape:output:0*

DstT0*

SrcT0	*
_class
loc:@R/Reshape*
_output_shapes
:V
gradient_tape/R/SizeConst*
_output_shapes
: *
dtype0*
value	B :	`
gradient_tape/R/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/R/ExpandDims
ExpandDimsgradient_tape/R/Size:output:0'gradient_tape/R/ExpandDims/dim:output:0*
T0*
_output_shapes
:W
gradient_tape/R/ConstConst*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/R/Const_1Const*
_output_shapes
: *
dtype0*
value	B :y
#gradient_tape/R/strided_slice/stackPackgradient_tape/R/Const:output:0*
N*
T0*
_output_shapes
:u
%gradient_tape/R/strided_slice/stack_1PackR/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:}
%gradient_tape/R/strided_slice/stack_2Pack gradient_tape/R/Const_1:output:0*
N*
T0*
_output_shapes
:
gradient_tape/R/strided_sliceStridedSlicegradient_tape/R/Cast:y:0,gradient_tape/R/strided_slice/stack:output:0.gradient_tape/R/strided_slice/stack_1:output:0.gradient_tape/R/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskY
gradient_tape/R/Const_2Const*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/R/Const_3Const*
_output_shapes
: *
dtype0*
value	B :u
%gradient_tape/R/strided_slice_1/stackPackR/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/R/strided_slice_1/stack_1Pack gradient_tape/R/Const_2:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/R/strided_slice_1/stack_2Pack gradient_tape/R/Const_3:output:0*
N*
T0*
_output_shapes
:
gradient_tape/R/strided_slice_1StridedSlicegradient_tape/R/Cast:y:0.gradient_tape/R/strided_slice_1/stack:output:00gradient_tape/R/strided_slice_1/stack_1:output:00gradient_tape/R/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_masko
%gradient_tape/R/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:q
'gradient_tape/R/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'gradient_tape/R/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
gradient_tape/R/strided_slice_2StridedSlice(gradient_tape/R/strided_slice_1:output:0.gradient_tape/R/strided_slice_2/stack:output:00gradient_tape/R/strided_slice_2/stack_1:output:00gradient_tape/R/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskr
gradient_tape/R/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ]
gradient_tape/R/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ђ
gradient_tape/R/concatConcatV2&gradient_tape/R/strided_slice:output:0(gradient_tape/R/concat/values_1:output:0(gradient_tape/R/strided_slice_2:output:0$gradient_tape/R/concat/axis:output:0*
N*
T0*
_output_shapes
:X
gradient_tape/R/Size_1Const*
_output_shapes
: *
dtype0*
value	B :X
gradient_tape/R/Size_2Const*
_output_shapes
: *
dtype0*
value	B :]
gradient_tape/R/range/startConst*
_output_shapes
: *
dtype0*
value	B : ]
gradient_tape/R/range/limitConst*
_output_shapes
: *
dtype0*
value	B : ]
gradient_tape/R/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Њ
gradient_tape/R/rangeRange$gradient_tape/R/range/start:output:0$gradient_tape/R/range/limit:output:0$gradient_tape/R/range/delta:output:0*
_output_shapes
: _
gradient_tape/R/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/R/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :­
gradient_tape/R/range_1Range&gradient_tape/R/range_1/start:output:0gradient_tape/R/Size_2:output:0&gradient_tape/R/range_1/delta:output:0*
_output_shapes
:W
gradient_tape/R/add/yConst*
_output_shapes
: *
dtype0*
value	B :~
gradient_tape/R/addAddV2gradient_tape/R/Size_2:output:0gradient_tape/R/add/y:output:0*
T0*
_output_shapes
: _
gradient_tape/R/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/R/range_2Rangegradient_tape/R/add:z:0gradient_tape/R/Size_1:output:0&gradient_tape/R/range_2/delta:output:0*
_output_shapes
: Њ
gradient_tape/R/ReshapeReshape&gradient_tape/A/einsum/Einsum:output:0gradient_tape/R/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџx
!gradient_tape/R/concat_1/values_1Packgradient_tape/R/Size_2:output:0*
N*
T0*
_output_shapes
:_
gradient_tape/R/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/R/concat_1ConcatV2gradient_tape/R/range:output:0*gradient_tape/R/concat_1/values_1:output:0 gradient_tape/R/range_1:output:0 gradient_tape/R/range_2:output:0&gradient_tape/R/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Њ
gradient_tape/R/transpose	Transpose gradient_tape/R/Reshape:output:0!gradient_tape/R/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ_
gradient_tape/R/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : б
gradient_tape/R/GatherV2GatherV2gradient_tape/R/Cast:y:0!gradient_tape/R/concat_1:output:0&gradient_tape/R/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
gradient_tape/R/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
gradient_tape/R/add_1AddV2R/GatherV2/axis:output:0 gradient_tape/R/add_1/y:output:0*
T0*
_output_shapes
: u
%gradient_tape/R/strided_slice_3/stackPackR/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:x
'gradient_tape/R/strided_slice_3/stack_1Packgradient_tape/R/add_1:z:0*
N*
T0*
_output_shapes
:q
'gradient_tape/R/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
gradient_tape/R/strided_slice_3StridedSlicegradient_tape/R/Cast:y:0.gradient_tape/R/strided_slice_3/stack:output:00gradient_tape/R/strided_slice_3/stack_1:output:00gradient_tape/R/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
gradient_tape/R/Size_3Const*
_output_shapes
: *
dtype0*
value	B :	b
 gradient_tape/R/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/R/ExpandDims_1
ExpandDimsgradient_tape/R/Size_3:output:0)gradient_tape/R/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:
gradient_tape/R/Reshape_1Reshaper_gatherv2_indices%gradient_tape/R/ExpandDims_1:output:0*
T0*
_output_shapes
:	є
"gradient_tape/R/UnsortedSegmentSumUnsortedSegmentSumgradient_tape/R/transpose:y:0"gradient_tape/R/Reshape_1:output:0(gradient_tape/R/strided_slice_3:output:0*
Tindices0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџY
gradient_tape/R/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/R/add_2AddV2 gradient_tape/R/range_1:output:0 gradient_tape/R/add_2/y:output:0*
T0*
_output_shapes
:k
!gradient_tape/R/concat_2/values_2Const*
_output_shapes
:*
dtype0*
valueB: _
gradient_tape/R/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/R/concat_2ConcatV2gradient_tape/R/range:output:0gradient_tape/R/add_2:z:0*gradient_tape/R/concat_2/values_2:output:0 gradient_tape/R/range_2:output:0&gradient_tape/R/concat_2/axis:output:0*
N*
T0*
_output_shapes
:З
gradient_tape/R/transpose_1	Transpose+gradient_tape/R/UnsortedSegmentSum:output:0!gradient_tape/R/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
gradient_tape/Y/mul_35/MulMul(gradient_tape/A/einsum/Einsum_1:output:0
y_mul_35_y*
T0*'
_output_shapes
:џџџџџџџџџ	y
(gradient_tape/ChemIndTransf/einsum/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      {
*gradient_tape/ChemIndTransf/einsum/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      
=gradient_tape/ChemIndTransf/einsum/Einsum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?gradient_tape/ChemIndTransf/einsum/Einsum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?gradient_tape/ChemIndTransf/einsum/Einsum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7gradient_tape/ChemIndTransf/einsum/Einsum/strided_sliceStridedSlicegradient_tape/A/Cast:y:0Fgradient_tape/ChemIndTransf/einsum/Einsum/strided_slice/stack:output:0Hgradient_tape/ChemIndTransf/einsum/Einsum/strided_slice/stack_1:output:0Hgradient_tape/ChemIndTransf/einsum/Einsum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
2gradient_tape/ChemIndTransf/einsum/Einsum/inputs_0UnsortedSegmentSum"gradient_tape/A/Reshape_1:output:0"gradient_tape/A/Reshape_2:output:0@gradient_tape/ChemIndTransf/einsum/Einsum/strided_slice:output:0*
Tindices0*
T0*
_output_shapes

:й
)gradient_tape/ChemIndTransf/einsum/EinsumEinsum;gradient_tape/ChemIndTransf/einsum/Einsum/inputs_0:output:0ChemIndTransf/mul:z:0*
N*
T0*
_output_shapes

:*
equation...n,...kn->...k
?gradient_tape/ChemIndTransf/einsum/Einsum_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Agradient_tape/ChemIndTransf/einsum/Einsum_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Agradient_tape/ChemIndTransf/einsum/Einsum_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
9gradient_tape/ChemIndTransf/einsum/Einsum_1/strided_sliceStridedSlicegradient_tape/A/Cast:y:0Hgradient_tape/ChemIndTransf/einsum/Einsum_1/strided_slice/stack:output:0Jgradient_tape/ChemIndTransf/einsum/Einsum_1/strided_slice/stack_1:output:0Jgradient_tape/ChemIndTransf/einsum/Einsum_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
4gradient_tape/ChemIndTransf/einsum/Einsum_1/inputs_0UnsortedSegmentSum"gradient_tape/A/Reshape_1:output:0"gradient_tape/A/Reshape_2:output:0Bgradient_tape/ChemIndTransf/einsum/Einsum_1/strided_slice:output:0*
Tindices0*
T0*
_output_shapes

:ў
+gradient_tape/ChemIndTransf/einsum/Einsum_1Einsum=gradient_tape/ChemIndTransf/einsum/Einsum_1/inputs_0:output:02ChemIndTransf/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*"
_output_shapes
:*
equation...n,...k->...kn
6gradient_tape/ChemIndTransf/einsum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8gradient_tape/ChemIndTransf/einsum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
8gradient_tape/ChemIndTransf/einsum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ь
0gradient_tape/ChemIndTransf/einsum/strided_sliceStridedSlice1gradient_tape/ChemIndTransf/einsum/Shape:output:0?gradient_tape/ChemIndTransf/einsum/strided_slice/stack:output:0Agradient_tape/ChemIndTransf/einsum/strided_slice/stack_1:output:0Agradient_tape/ChemIndTransf/einsum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:
8gradient_tape/ChemIndTransf/einsum/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:gradient_tape/ChemIndTransf/einsum/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ
:gradient_tape/ChemIndTransf/einsum/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
2gradient_tape/ChemIndTransf/einsum/strided_slice_1StridedSlice3gradient_tape/ChemIndTransf/einsum/Shape_1:output:0Agradient_tape/ChemIndTransf/einsum/strided_slice_1/stack:output:0Cgradient_tape/ChemIndTransf/einsum/strided_slice_1/stack_1:output:0Cgradient_tape/ChemIndTransf/einsum/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: §
8gradient_tape/ChemIndTransf/einsum/BroadcastGradientArgsBroadcastGradientArgs9gradient_tape/ChemIndTransf/einsum/strided_slice:output:0;gradient_tape/ChemIndTransf/einsum/strided_slice_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџj
(gradient_tape/ChemIndTransf/einsum/add/xConst*
_output_shapes
: *
dtype0*
value	B : Я
&gradient_tape/ChemIndTransf/einsum/addAddV21gradient_tape/ChemIndTransf/einsum/add/x:output:0=gradient_tape/ChemIndTransf/einsum/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџЖ
&gradient_tape/ChemIndTransf/einsum/SumSum2gradient_tape/ChemIndTransf/einsum/Einsum:output:0*gradient_tape/ChemIndTransf/einsum/add:z:0*
T0*
_output_shapes

:Т
*gradient_tape/ChemIndTransf/einsum/ReshapeReshape/gradient_tape/ChemIndTransf/einsum/Sum:output:01gradient_tape/ChemIndTransf/einsum/Shape:output:0*
T0*
_output_shapes

:l
*gradient_tape/ChemIndTransf/einsum/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : г
(gradient_tape/ChemIndTransf/einsum/add_1AddV23gradient_tape/ChemIndTransf/einsum/add_1/x:output:0=gradient_tape/ChemIndTransf/einsum/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџМ
(gradient_tape/ChemIndTransf/einsum/Sum_1Sum4gradient_tape/ChemIndTransf/einsum/Einsum_1:output:0,gradient_tape/ChemIndTransf/einsum/add_1:z:0*
T0*
_output_shapes

:Ш
,gradient_tape/ChemIndTransf/einsum/Reshape_1Reshape1gradient_tape/ChemIndTransf/einsum/Sum_1:output:03gradient_tape/ChemIndTransf/einsum/Shape_1:output:0*
T0*
_output_shapes

:x
gradient_tape/R/Shape_1Shape#DenseLayer/einsum_3/Einsum:output:0*
T0*
_output_shapes
::эЯ
gradient_tape/R/Reshape_2Reshapegradient_tape/R/transpose_1:y:0 gradient_tape/R/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџw
+gradient_tape/Y/transpose/InvertPermutationInvertPermutationY/transpose/perm:output:0*
_output_shapes
:Г
#gradient_tape/Y/transpose/transpose	Transposegradient_tape/Y/mul_35/Mul:z:0/gradient_tape/Y/transpose/InvertPermutation:y:0*
T0*'
_output_shapes
:	џџџџџџџџџ
#gradient_tape/ChemIndTransf/mul/MulMul5gradient_tape/ChemIndTransf/einsum/Reshape_1:output:0chemindtransf_mul_y*
T0*
_output_shapes

:z
'gradient_tape/DenseLayer/einsum_3/ShapeShapeDenseLayer/mul_11:z:0*
T0*
_output_shapes
::эЯz
)gradient_tape/DenseLayer/einsum_3/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"@      Ш
(gradient_tape/DenseLayer/einsum_3/EinsumEinsum"gradient_tape/R/Reshape_2:output:0DenseLayer/mul_12:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@*
equation...n,...kn->...kЮ
*gradient_tape/DenseLayer/einsum_3/Einsum_1Einsum"gradient_tape/R/Reshape_2:output:0DenseLayer/mul_11:z:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ@*
equation...n,...k->...kn
5gradient_tape/DenseLayer/einsum_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7gradient_tape/DenseLayer/einsum_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
7gradient_tape/DenseLayer/einsum_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
/gradient_tape/DenseLayer/einsum_3/strided_sliceStridedSlice0gradient_tape/DenseLayer/einsum_3/Shape:output:0>gradient_tape/DenseLayer/einsum_3/strided_slice/stack:output:0@gradient_tape/DenseLayer/einsum_3/strided_slice/stack_1:output:0@gradient_tape/DenseLayer/einsum_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:
7gradient_tape/DenseLayer/einsum_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9gradient_tape/DenseLayer/einsum_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ
9gradient_tape/DenseLayer/einsum_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
1gradient_tape/DenseLayer/einsum_3/strided_slice_1StridedSlice2gradient_tape/DenseLayer/einsum_3/Shape_1:output:0@gradient_tape/DenseLayer/einsum_3/strided_slice_1/stack:output:0Bgradient_tape/DenseLayer/einsum_3/strided_slice_1/stack_1:output:0Bgradient_tape/DenseLayer/einsum_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: њ
7gradient_tape/DenseLayer/einsum_3/BroadcastGradientArgsBroadcastGradientArgs8gradient_tape/DenseLayer/einsum_3/strided_slice:output:0:gradient_tape/DenseLayer/einsum_3/strided_slice_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџi
'gradient_tape/DenseLayer/einsum_3/add/xConst*
_output_shapes
: *
dtype0*
value	B : Ь
%gradient_tape/DenseLayer/einsum_3/addAddV20gradient_tape/DenseLayer/einsum_3/add/x:output:0<gradient_tape/DenseLayer/einsum_3/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ­
%gradient_tape/DenseLayer/einsum_3/SumSum1gradient_tape/DenseLayer/einsum_3/Einsum:output:0)gradient_tape/DenseLayer/einsum_3/add:z:0*
T0*
_output_shapes
:Ш
)gradient_tape/DenseLayer/einsum_3/ReshapeReshape.gradient_tape/DenseLayer/einsum_3/Sum:output:00gradient_tape/DenseLayer/einsum_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@k
)gradient_tape/DenseLayer/einsum_3/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : а
'gradient_tape/DenseLayer/einsum_3/add_1AddV22gradient_tape/DenseLayer/einsum_3/add_1/x:output:0<gradient_tape/DenseLayer/einsum_3/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџГ
'gradient_tape/DenseLayer/einsum_3/Sum_1Sum3gradient_tape/DenseLayer/einsum_3/Einsum_1:output:0+gradient_tape/DenseLayer/einsum_3/add_1:z:0*
T0*
_output_shapes
:Х
+gradient_tape/DenseLayer/einsum_3/Reshape_1Reshape0gradient_tape/DenseLayer/einsum_3/Sum_1:output:02gradient_tape/DenseLayer/einsum_3/Shape_1:output:0*
T0*
_output_shapes

:@
gradient_tape/Y/stack_1/unstackUnpack'gradient_tape/Y/transpose/transpose:y:0*
T0*
_output_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*	
num	ж
#gradient_tape/DenseLayer/mul_11/MulMul2gradient_tape/DenseLayer/einsum_3/Reshape:output:0DenseLayer/mul_11/y:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ@
#gradient_tape/DenseLayer/mul_12/MulMul4gradient_tape/DenseLayer/einsum_3/Reshape_1:output:0denselayer_mul_12_y*
T0*
_output_shapes

:@s
%gradient_tape/Y/strided_slice_6/ShapeShapeY/stack:output:0*
T0*
_output_shapes
::эЯ
6gradient_tape/Y/strided_slice_6/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB: ~
4gradient_tape/Y/strided_slice_6/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB:
8gradient_tape/Y/strided_slice_6/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:И
0gradient_tape/Y/strided_slice_6/StridedSliceGradStridedSliceGrad.gradient_tape/Y/strided_slice_6/Shape:output:0?gradient_tape/Y/strided_slice_6/StridedSliceGrad/begin:output:0=gradient_tape/Y/strided_slice_6/StridedSliceGrad/end:output:0Agradient_tape/Y/strided_slice_6/StridedSliceGrad/strides:output:0(gradient_tape/Y/stack_1/unstack:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gradient_tape/Y/mul_24/MulMul(gradient_tape/Y/stack_1/unstack:output:1Y/SelectV2_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџs
%gradient_tape/Y/strided_slice_8/ShapeShapeY/stack:output:0*
T0*
_output_shapes
::эЯ
6gradient_tape/Y/strided_slice_8/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:~
4gradient_tape/Y/strided_slice_8/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB:
8gradient_tape/Y/strided_slice_8/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:И
0gradient_tape/Y/strided_slice_8/StridedSliceGradStridedSliceGrad.gradient_tape/Y/strided_slice_8/Shape:output:0?gradient_tape/Y/strided_slice_8/StridedSliceGrad/begin:output:0=gradient_tape/Y/strided_slice_8/StridedSliceGrad/end:output:0Agradient_tape/Y/strided_slice_8/StridedSliceGrad/strides:output:0(gradient_tape/Y/stack_1/unstack:output:2*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gradient_tape/Y/mul_26/MulMul(gradient_tape/Y/stack_1/unstack:output:3Y/SelectV2_3:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_28/MulMul(gradient_tape/Y/stack_1/unstack:output:4Y/SelectV2_4:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_30/MulMul(gradient_tape/Y/stack_1/unstack:output:5Y/SelectV2_5:output:0*
T0*#
_output_shapes
:џџџџџџџџџt
&gradient_tape/Y/strided_slice_10/ShapeShapeY/stack:output:0*
T0*
_output_shapes
::эЯ
7gradient_tape/Y/strided_slice_10/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
5gradient_tape/Y/strided_slice_10/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB:
9gradient_tape/Y/strided_slice_10/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:Н
1gradient_tape/Y/strided_slice_10/StridedSliceGradStridedSliceGrad/gradient_tape/Y/strided_slice_10/Shape:output:0@gradient_tape/Y/strided_slice_10/StridedSliceGrad/begin:output:0>gradient_tape/Y/strided_slice_10/StridedSliceGrad/end:output:0Bgradient_tape/Y/strided_slice_10/StridedSliceGrad/strides:output:0(gradient_tape/Y/stack_1/unstack:output:6*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gradient_tape/Y/mul_32/MulMul(gradient_tape/Y/stack_1/unstack:output:7Y/SelectV2_7:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_34/MulMul(gradient_tape/Y/stack_1/unstack:output:8Y/SelectV2_8:output:0*
T0*#
_output_shapes
:џџџџџџџџџl
zeros_like_1	ZerosLikeDenseLayer/IdentityN_2:output:1*
T0*'
_output_shapes
:џџџџџџџџџ@P
zeros_1Const*
_output_shapes
: *
dtype0*
valueB 2        Ё
mul_11MulDenseLayer/Cast_2:y:0#DenseLayer/einsum_2/Einsum:output:0$^gradient_tape/DenseLayer/mul_11/Mul*
T0*'
_output_shapes
:џџџџџџџџџ@R
	Sigmoid_1Sigmoid
mul_11:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@{
mul_12MulDenseLayer/Cast_2:y:0#DenseLayer/einsum_2/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?_
sub_2Subsub_2/x:output:0Sigmoid_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_13Mul
mul_12:z:0	sub_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
add_5/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?^
add_5AddV2add_5/x:output:0
mul_13:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_14MulSigmoid_1:y:0	add_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@i
Square_1Square#DenseLayer/einsum_2/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@v
mul_15Mul'gradient_tape/DenseLayer/mul_11/Mul:z:0Square_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_16Mul
mul_15:z:0Sigmoid_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?_
sub_3Subsub_3/x:output:0Sigmoid_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_17Mul
mul_16:z:0	sub_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@X
Const_2Const*
_output_shapes
:*
dtype0*
valueB"       K
Sum_1Sum
mul_17:z:0Const_2:output:0*
T0*
_output_shapes
: t
mul_18Mul'gradient_tape/DenseLayer/mul_11/Mul:z:0
mul_14:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@}
gradient_tape/Y/mul_23/MulMulY/Sqrt_5:y:0gradient_tape/Y/mul_24/Mul:z:0*
T0*#
_output_shapes
:џџџџџџџџџ_
gradient_tape/Y/mul_23/ShapeConst*
_output_shapes
: *
dtype0*
valueB h
gradient_tape/Y/mul_23/Shape_1ShapeY/mul_14:z:0*
T0*
_output_shapes
::эЯ}
gradient_tape/Y/mul_25/MulMulY/Sqrt_7:y:0gradient_tape/Y/mul_26/Mul:z:0*
T0*#
_output_shapes
:џџџџџџџџџ_
gradient_tape/Y/mul_25/ShapeConst*
_output_shapes
: *
dtype0*
valueB h
gradient_tape/Y/mul_25/Shape_1ShapeY/mul_13:z:0*
T0*
_output_shapes
::эЯ}
gradient_tape/Y/mul_27/MulMulY/Sqrt_8:y:0gradient_tape/Y/mul_28/Mul:z:0*
T0*#
_output_shapes
:џџџџџџџџџ_
gradient_tape/Y/mul_27/ShapeConst*
_output_shapes
: *
dtype0*
valueB h
gradient_tape/Y/mul_27/Shape_1ShapeY/mul_22:z:0*
T0*
_output_shapes
::эЯ}
gradient_tape/Y/mul_29/MulMulY/Sqrt_9:y:0gradient_tape/Y/mul_30/Mul:z:0*
T0*#
_output_shapes
:џџџџџџџџџ_
gradient_tape/Y/mul_29/ShapeConst*
_output_shapes
: *
dtype0*
valueB h
gradient_tape/Y/mul_29/Shape_1ShapeY/mul_16:z:0*
T0*
_output_shapes
::эЯ~
gradient_tape/Y/mul_31/MulMulY/Sqrt_11:y:0gradient_tape/Y/mul_32/Mul:z:0*
T0*#
_output_shapes
:џџџџџџџџџ_
gradient_tape/Y/mul_31/ShapeConst*
_output_shapes
: *
dtype0*
valueB h
gradient_tape/Y/mul_31/Shape_1ShapeY/mul_15:z:0*
T0*
_output_shapes
::эЯ~
gradient_tape/Y/mul_33/MulMulY/Sqrt_12:y:0gradient_tape/Y/mul_34/Mul:z:0*
T0*#
_output_shapes
:џџџџџџџџџ_
gradient_tape/Y/mul_33/ShapeConst*
_output_shapes
: *
dtype0*
valueB h
gradient_tape/Y/mul_33/Shape_1ShapeY/mul_21:z:0*
T0*
_output_shapes
::эЯ
gradient_tape/Y/mul_14/MulMulgradient_tape/Y/mul_23/Mul:z:0Y/strided_slice_13:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_14/Mul_1MulY/strided_slice_1:output:0gradient_tape/Y/mul_23/Mul:z:0*
T0*#
_output_shapes
:џџџџџџџџџt
gradient_tape/Y/mul_14/ShapeShapeY/strided_slice_1:output:0*
T0*
_output_shapes
::эЯw
gradient_tape/Y/mul_14/Shape_1ShapeY/strided_slice_13:output:0*
T0*
_output_shapes
::эЯЩ
,gradient_tape/Y/mul_14/BroadcastGradientArgsBroadcastGradientArgs%gradient_tape/Y/mul_14/Shape:output:0'gradient_tape/Y/mul_14/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџГ
gradient_tape/Y/mul_14/SumSumgradient_tape/Y/mul_14/Mul:z:01gradient_tape/Y/mul_14/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Ѓ
gradient_tape/Y/mul_14/ReshapeReshape#gradient_tape/Y/mul_14/Sum:output:0%gradient_tape/Y/mul_14/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџЗ
gradient_tape/Y/mul_14/Sum_1Sum gradient_tape/Y/mul_14/Mul_1:z:01gradient_tape/Y/mul_14/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Љ
 gradient_tape/Y/mul_14/Reshape_1Reshape%gradient_tape/Y/mul_14/Sum_1:output:0'gradient_tape/Y/mul_14/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_13/MulMulgradient_tape/Y/mul_25/Mul:z:0Y/strided_slice_12:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_13/Mul_1MulY/strided_slice:output:0gradient_tape/Y/mul_25/Mul:z:0*
T0*#
_output_shapes
:џџџџџџџџџr
gradient_tape/Y/mul_13/ShapeShapeY/strided_slice:output:0*
T0*
_output_shapes
::эЯw
gradient_tape/Y/mul_13/Shape_1ShapeY/strided_slice_12:output:0*
T0*
_output_shapes
::эЯЩ
,gradient_tape/Y/mul_13/BroadcastGradientArgsBroadcastGradientArgs%gradient_tape/Y/mul_13/Shape:output:0'gradient_tape/Y/mul_13/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџГ
gradient_tape/Y/mul_13/SumSumgradient_tape/Y/mul_13/Mul:z:01gradient_tape/Y/mul_13/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Ѓ
gradient_tape/Y/mul_13/ReshapeReshape#gradient_tape/Y/mul_13/Sum:output:0%gradient_tape/Y/mul_13/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџЗ
gradient_tape/Y/mul_13/Sum_1Sum gradient_tape/Y/mul_13/Mul_1:z:01gradient_tape/Y/mul_13/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Љ
 gradient_tape/Y/mul_13/Reshape_1Reshape%gradient_tape/Y/mul_13/Sum_1:output:0'gradient_tape/Y/mul_13/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_22/MulMulgradient_tape/Y/mul_27/Mul:z:0Y/strided_slice_17:output:0*
T0*#
_output_shapes
:џџџџџџџџџ~
gradient_tape/Y/mul_22/Mul_1MulY/add_3:z:0gradient_tape/Y/mul_27/Mul:z:0*
T0*#
_output_shapes
:џџџџџџџџџe
gradient_tape/Y/mul_22/ShapeShapeY/add_3:z:0*
T0*
_output_shapes
::эЯw
gradient_tape/Y/mul_22/Shape_1ShapeY/strided_slice_17:output:0*
T0*
_output_shapes
::эЯЩ
,gradient_tape/Y/mul_22/BroadcastGradientArgsBroadcastGradientArgs%gradient_tape/Y/mul_22/Shape:output:0'gradient_tape/Y/mul_22/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџГ
gradient_tape/Y/mul_22/SumSumgradient_tape/Y/mul_22/Mul:z:01gradient_tape/Y/mul_22/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Ѓ
gradient_tape/Y/mul_22/ReshapeReshape#gradient_tape/Y/mul_22/Sum:output:0%gradient_tape/Y/mul_22/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџЗ
gradient_tape/Y/mul_22/Sum_1Sum gradient_tape/Y/mul_22/Mul_1:z:01gradient_tape/Y/mul_22/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Љ
 gradient_tape/Y/mul_22/Reshape_1Reshape%gradient_tape/Y/mul_22/Sum_1:output:0'gradient_tape/Y/mul_22/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_16/MulMulgradient_tape/Y/mul_29/Mul:z:0Y/strided_slice_15:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_16/Mul_1MulY/strided_slice_1:output:0gradient_tape/Y/mul_29/Mul:z:0*
T0*#
_output_shapes
:џџџџџџџџџt
gradient_tape/Y/mul_16/ShapeShapeY/strided_slice_1:output:0*
T0*
_output_shapes
::эЯw
gradient_tape/Y/mul_16/Shape_1ShapeY/strided_slice_15:output:0*
T0*
_output_shapes
::эЯЩ
,gradient_tape/Y/mul_16/BroadcastGradientArgsBroadcastGradientArgs%gradient_tape/Y/mul_16/Shape:output:0'gradient_tape/Y/mul_16/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџГ
gradient_tape/Y/mul_16/SumSumgradient_tape/Y/mul_16/Mul:z:01gradient_tape/Y/mul_16/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Ѓ
gradient_tape/Y/mul_16/ReshapeReshape#gradient_tape/Y/mul_16/Sum:output:0%gradient_tape/Y/mul_16/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџЗ
gradient_tape/Y/mul_16/Sum_1Sum gradient_tape/Y/mul_16/Mul_1:z:01gradient_tape/Y/mul_16/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Љ
 gradient_tape/Y/mul_16/Reshape_1Reshape%gradient_tape/Y/mul_16/Sum_1:output:0'gradient_tape/Y/mul_16/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_15/MulMulgradient_tape/Y/mul_31/Mul:z:0Y/strided_slice_14:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_15/Mul_1MulY/strided_slice:output:0gradient_tape/Y/mul_31/Mul:z:0*
T0*#
_output_shapes
:џџџџџџџџџr
gradient_tape/Y/mul_15/ShapeShapeY/strided_slice:output:0*
T0*
_output_shapes
::эЯw
gradient_tape/Y/mul_15/Shape_1ShapeY/strided_slice_14:output:0*
T0*
_output_shapes
::эЯЩ
,gradient_tape/Y/mul_15/BroadcastGradientArgsBroadcastGradientArgs%gradient_tape/Y/mul_15/Shape:output:0'gradient_tape/Y/mul_15/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџГ
gradient_tape/Y/mul_15/SumSumgradient_tape/Y/mul_15/Mul:z:01gradient_tape/Y/mul_15/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Ѓ
gradient_tape/Y/mul_15/ReshapeReshape#gradient_tape/Y/mul_15/Sum:output:0%gradient_tape/Y/mul_15/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџЗ
gradient_tape/Y/mul_15/Sum_1Sum gradient_tape/Y/mul_15/Mul_1:z:01gradient_tape/Y/mul_15/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Љ
 gradient_tape/Y/mul_15/Reshape_1Reshape%gradient_tape/Y/mul_15/Sum_1:output:0'gradient_tape/Y/mul_15/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_21/MulMulgradient_tape/Y/mul_33/Mul:z:0Y/strided_slice_16:output:0*
T0*#
_output_shapes
:џџџџџџџџџ~
gradient_tape/Y/mul_21/Mul_1MulY/sub_1:z:0gradient_tape/Y/mul_33/Mul:z:0*
T0*#
_output_shapes
:џџџџџџџџџe
gradient_tape/Y/mul_21/ShapeShapeY/sub_1:z:0*
T0*
_output_shapes
::эЯw
gradient_tape/Y/mul_21/Shape_1ShapeY/strided_slice_16:output:0*
T0*
_output_shapes
::эЯЩ
,gradient_tape/Y/mul_21/BroadcastGradientArgsBroadcastGradientArgs%gradient_tape/Y/mul_21/Shape:output:0'gradient_tape/Y/mul_21/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџГ
gradient_tape/Y/mul_21/SumSumgradient_tape/Y/mul_21/Mul:z:01gradient_tape/Y/mul_21/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Ѓ
gradient_tape/Y/mul_21/ReshapeReshape#gradient_tape/Y/mul_21/Sum:output:0%gradient_tape/Y/mul_21/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџЗ
gradient_tape/Y/mul_21/Sum_1Sum gradient_tape/Y/mul_21/Mul_1:z:01gradient_tape/Y/mul_21/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Љ
 gradient_tape/Y/mul_21/Reshape_1Reshape%gradient_tape/Y/mul_21/Sum_1:output:0'gradient_tape/Y/mul_21/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџt
&gradient_tape/Y/strided_slice_13/ShapeShapeY/stack:output:0*
T0*
_output_shapes
::эЯ
7gradient_tape/Y/strided_slice_13/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
5gradient_tape/Y/strided_slice_13/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB:
9gradient_tape/Y/strided_slice_13/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:О
1gradient_tape/Y/strided_slice_13/StridedSliceGradStridedSliceGrad/gradient_tape/Y/strided_slice_13/Shape:output:0@gradient_tape/Y/strided_slice_13/StridedSliceGrad/begin:output:0>gradient_tape/Y/strided_slice_13/StridedSliceGrad/end:output:0Bgradient_tape/Y/strided_slice_13/StridedSliceGrad/strides:output:0)gradient_tape/Y/mul_14/Reshape_1:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskt
&gradient_tape/Y/strided_slice_12/ShapeShapeY/stack:output:0*
T0*
_output_shapes
::эЯ
7gradient_tape/Y/strided_slice_12/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
5gradient_tape/Y/strided_slice_12/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB:
9gradient_tape/Y/strided_slice_12/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:О
1gradient_tape/Y/strided_slice_12/StridedSliceGradStridedSliceGrad/gradient_tape/Y/strided_slice_12/Shape:output:0@gradient_tape/Y/strided_slice_12/StridedSliceGrad/begin:output:0>gradient_tape/Y/strided_slice_12/StridedSliceGrad/end:output:0Bgradient_tape/Y/strided_slice_12/StridedSliceGrad/strides:output:0)gradient_tape/Y/mul_13/Reshape_1:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
gradient_tape/Y/add_3/ShapeShapeY/mul_19:z:0*
T0*
_output_shapes
::эЯg
gradient_tape/Y/add_3/Shape_1ShapeY/mul_20:z:0*
T0*
_output_shapes
::эЯЦ
+gradient_tape/Y/add_3/BroadcastGradientArgsBroadcastGradientArgs$gradient_tape/Y/add_3/Shape:output:0&gradient_tape/Y/add_3/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџК
gradient_tape/Y/add_3/SumSum'gradient_tape/Y/mul_22/Reshape:output:00gradient_tape/Y/add_3/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
gradient_tape/Y/add_3/ReshapeReshape"gradient_tape/Y/add_3/Sum:output:0$gradient_tape/Y/add_3/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџМ
gradient_tape/Y/add_3/Sum_1Sum'gradient_tape/Y/mul_22/Reshape:output:00gradient_tape/Y/add_3/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(І
gradient_tape/Y/add_3/Reshape_1Reshape$gradient_tape/Y/add_3/Sum_1:output:0&gradient_tape/Y/add_3/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџt
&gradient_tape/Y/strided_slice_17/ShapeShapeY/stack:output:0*
T0*
_output_shapes
::эЯ
7gradient_tape/Y/strided_slice_17/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
5gradient_tape/Y/strided_slice_17/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB:
9gradient_tape/Y/strided_slice_17/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:О
1gradient_tape/Y/strided_slice_17/StridedSliceGradStridedSliceGrad/gradient_tape/Y/strided_slice_17/Shape:output:0@gradient_tape/Y/strided_slice_17/StridedSliceGrad/begin:output:0>gradient_tape/Y/strided_slice_17/StridedSliceGrad/end:output:0Bgradient_tape/Y/strided_slice_17/StridedSliceGrad/strides:output:0)gradient_tape/Y/mul_22/Reshape_1:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskt
&gradient_tape/Y/strided_slice_15/ShapeShapeY/stack:output:0*
T0*
_output_shapes
::эЯ
7gradient_tape/Y/strided_slice_15/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
5gradient_tape/Y/strided_slice_15/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB:
9gradient_tape/Y/strided_slice_15/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:О
1gradient_tape/Y/strided_slice_15/StridedSliceGradStridedSliceGrad/gradient_tape/Y/strided_slice_15/Shape:output:0@gradient_tape/Y/strided_slice_15/StridedSliceGrad/begin:output:0>gradient_tape/Y/strided_slice_15/StridedSliceGrad/end:output:0Bgradient_tape/Y/strided_slice_15/StridedSliceGrad/strides:output:0)gradient_tape/Y/mul_16/Reshape_1:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskt
&gradient_tape/Y/strided_slice_14/ShapeShapeY/stack:output:0*
T0*
_output_shapes
::эЯ
7gradient_tape/Y/strided_slice_14/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
5gradient_tape/Y/strided_slice_14/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB:
9gradient_tape/Y/strided_slice_14/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:О
1gradient_tape/Y/strided_slice_14/StridedSliceGradStridedSliceGrad/gradient_tape/Y/strided_slice_14/Shape:output:0@gradient_tape/Y/strided_slice_14/StridedSliceGrad/begin:output:0>gradient_tape/Y/strided_slice_14/StridedSliceGrad/end:output:0Bgradient_tape/Y/strided_slice_14/StridedSliceGrad/strides:output:0)gradient_tape/Y/mul_15/Reshape_1:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskw
gradient_tape/Y/sub_1/NegNeg'gradient_tape/Y/mul_21/Reshape:output:0*
T0*#
_output_shapes
:џџџџџџџџџe
gradient_tape/Y/sub_1/ShapeShapeY/mul_17:z:0*
T0*
_output_shapes
::эЯg
gradient_tape/Y/sub_1/Shape_1ShapeY/mul_18:z:0*
T0*
_output_shapes
::эЯЦ
+gradient_tape/Y/sub_1/BroadcastGradientArgsBroadcastGradientArgs$gradient_tape/Y/sub_1/Shape:output:0&gradient_tape/Y/sub_1/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџК
gradient_tape/Y/sub_1/SumSum'gradient_tape/Y/mul_21/Reshape:output:00gradient_tape/Y/sub_1/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
gradient_tape/Y/sub_1/ReshapeReshape"gradient_tape/Y/sub_1/Sum:output:0$gradient_tape/Y/sub_1/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџВ
gradient_tape/Y/sub_1/Sum_1Sumgradient_tape/Y/sub_1/Neg:y:00gradient_tape/Y/sub_1/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(І
gradient_tape/Y/sub_1/Reshape_1Reshape$gradient_tape/Y/sub_1/Sum_1:output:0&gradient_tape/Y/sub_1/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџt
&gradient_tape/Y/strided_slice_16/ShapeShapeY/stack:output:0*
T0*
_output_shapes
::эЯ
7gradient_tape/Y/strided_slice_16/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
5gradient_tape/Y/strided_slice_16/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB:
9gradient_tape/Y/strided_slice_16/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:О
1gradient_tape/Y/strided_slice_16/StridedSliceGradStridedSliceGrad/gradient_tape/Y/strided_slice_16/Shape:output:0@gradient_tape/Y/strided_slice_16/StridedSliceGrad/begin:output:0>gradient_tape/Y/strided_slice_16/StridedSliceGrad/end:output:0Bgradient_tape/Y/strided_slice_16/StridedSliceGrad/strides:output:0)gradient_tape/Y/mul_21/Reshape_1:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
gradient_tape/Y/mul_19/MulMul&gradient_tape/Y/add_3/Reshape:output:0Y/strided_slice_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_19/Mul_1MulY/strided_slice:output:0&gradient_tape/Y/add_3/Reshape:output:0*
T0*#
_output_shapes
:џџџџџџџџџr
gradient_tape/Y/mul_19/ShapeShapeY/strided_slice:output:0*
T0*
_output_shapes
::эЯv
gradient_tape/Y/mul_19/Shape_1ShapeY/strided_slice_1:output:0*
T0*
_output_shapes
::эЯЩ
,gradient_tape/Y/mul_19/BroadcastGradientArgsBroadcastGradientArgs%gradient_tape/Y/mul_19/Shape:output:0'gradient_tape/Y/mul_19/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџГ
gradient_tape/Y/mul_19/SumSumgradient_tape/Y/mul_19/Mul:z:01gradient_tape/Y/mul_19/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Ѓ
gradient_tape/Y/mul_19/ReshapeReshape#gradient_tape/Y/mul_19/Sum:output:0%gradient_tape/Y/mul_19/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџЗ
gradient_tape/Y/mul_19/Sum_1Sum gradient_tape/Y/mul_19/Mul_1:z:01gradient_tape/Y/mul_19/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Љ
 gradient_tape/Y/mul_19/Reshape_1Reshape%gradient_tape/Y/mul_19/Sum_1:output:0'gradient_tape/Y/mul_19/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_20/MulMul(gradient_tape/Y/add_3/Reshape_1:output:0Y/strided_slice:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_20/Mul_1MulY/strided_slice_1:output:0(gradient_tape/Y/add_3/Reshape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџt
gradient_tape/Y/mul_20/ShapeShapeY/strided_slice_1:output:0*
T0*
_output_shapes
::эЯt
gradient_tape/Y/mul_20/Shape_1ShapeY/strided_slice:output:0*
T0*
_output_shapes
::эЯЩ
,gradient_tape/Y/mul_20/BroadcastGradientArgsBroadcastGradientArgs%gradient_tape/Y/mul_20/Shape:output:0'gradient_tape/Y/mul_20/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџГ
gradient_tape/Y/mul_20/SumSumgradient_tape/Y/mul_20/Mul:z:01gradient_tape/Y/mul_20/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Ѓ
gradient_tape/Y/mul_20/ReshapeReshape#gradient_tape/Y/mul_20/Sum:output:0%gradient_tape/Y/mul_20/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџЗ
gradient_tape/Y/mul_20/Sum_1Sum gradient_tape/Y/mul_20/Mul_1:z:01gradient_tape/Y/mul_20/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Љ
 gradient_tape/Y/mul_20/Reshape_1Reshape%gradient_tape/Y/mul_20/Sum_1:output:0'gradient_tape/Y/mul_20/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_17/MulMul&gradient_tape/Y/sub_1/Reshape:output:0Y/strided_slice:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_17/Mul_1MulY/strided_slice:output:0&gradient_tape/Y/sub_1/Reshape:output:0*
T0*#
_output_shapes
:џџџџџџџџџr
gradient_tape/Y/mul_17/ShapeShapeY/strided_slice:output:0*
T0*
_output_shapes
::эЯt
gradient_tape/Y/mul_17/Shape_1ShapeY/strided_slice:output:0*
T0*
_output_shapes
::эЯЩ
,gradient_tape/Y/mul_17/BroadcastGradientArgsBroadcastGradientArgs%gradient_tape/Y/mul_17/Shape:output:0'gradient_tape/Y/mul_17/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџГ
gradient_tape/Y/mul_17/SumSumgradient_tape/Y/mul_17/Mul:z:01gradient_tape/Y/mul_17/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Ѓ
gradient_tape/Y/mul_17/ReshapeReshape#gradient_tape/Y/mul_17/Sum:output:0%gradient_tape/Y/mul_17/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџЗ
gradient_tape/Y/mul_17/Sum_1Sum gradient_tape/Y/mul_17/Mul_1:z:01gradient_tape/Y/mul_17/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Љ
 gradient_tape/Y/mul_17/Reshape_1Reshape%gradient_tape/Y/mul_17/Sum_1:output:0'gradient_tape/Y/mul_17/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_18/MulMul(gradient_tape/Y/sub_1/Reshape_1:output:0Y/strided_slice_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_18/Mul_1MulY/strided_slice_1:output:0(gradient_tape/Y/sub_1/Reshape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџt
gradient_tape/Y/mul_18/ShapeShapeY/strided_slice_1:output:0*
T0*
_output_shapes
::эЯv
gradient_tape/Y/mul_18/Shape_1ShapeY/strided_slice_1:output:0*
T0*
_output_shapes
::эЯЩ
,gradient_tape/Y/mul_18/BroadcastGradientArgsBroadcastGradientArgs%gradient_tape/Y/mul_18/Shape:output:0'gradient_tape/Y/mul_18/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџГ
gradient_tape/Y/mul_18/SumSumgradient_tape/Y/mul_18/Mul:z:01gradient_tape/Y/mul_18/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Ѓ
gradient_tape/Y/mul_18/ReshapeReshape#gradient_tape/Y/mul_18/Sum:output:0%gradient_tape/Y/mul_18/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџЗ
gradient_tape/Y/mul_18/Sum_1Sum gradient_tape/Y/mul_18/Mul_1:z:01gradient_tape/Y/mul_18/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Љ
 gradient_tape/Y/mul_18/Reshape_1Reshape%gradient_tape/Y/mul_18/Sum_1:output:0'gradient_tape/Y/mul_18/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџу
AddN_1AddN9gradient_tape/Y/strided_slice_6/StridedSliceGrad:output:09gradient_tape/Y/strided_slice_8/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_10/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_13/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_12/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_17/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_15/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_14/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_16/StridedSliceGrad:output:0*
N	*
T0*'
_output_shapes
:џџџџџџџџџЙ
gradient_tape/Y/stack/unstackUnpackAddN_1:sum:0*
T0*n
_output_shapes\
Z:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*	
numy
'gradient_tape/DenseLayer/einsum_2/ShapeShapeDenseLayer/mul_7:z:0*
T0*
_output_shapes
::эЯz
)gradient_tape/DenseLayer/einsum_2/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"@   @   Џ
(gradient_tape/DenseLayer/einsum_2/EinsumEinsum
mul_18:z:0DenseLayer/mul_8:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@*
equation...n,...kn->...kЕ
*gradient_tape/DenseLayer/einsum_2/Einsum_1Einsum
mul_18:z:0DenseLayer/mul_7:z:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ@@*
equation...n,...k->...kn
5gradient_tape/DenseLayer/einsum_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7gradient_tape/DenseLayer/einsum_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
7gradient_tape/DenseLayer/einsum_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
/gradient_tape/DenseLayer/einsum_2/strided_sliceStridedSlice0gradient_tape/DenseLayer/einsum_2/Shape:output:0>gradient_tape/DenseLayer/einsum_2/strided_slice/stack:output:0@gradient_tape/DenseLayer/einsum_2/strided_slice/stack_1:output:0@gradient_tape/DenseLayer/einsum_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:
7gradient_tape/DenseLayer/einsum_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9gradient_tape/DenseLayer/einsum_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ
9gradient_tape/DenseLayer/einsum_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
1gradient_tape/DenseLayer/einsum_2/strided_slice_1StridedSlice2gradient_tape/DenseLayer/einsum_2/Shape_1:output:0@gradient_tape/DenseLayer/einsum_2/strided_slice_1/stack:output:0Bgradient_tape/DenseLayer/einsum_2/strided_slice_1/stack_1:output:0Bgradient_tape/DenseLayer/einsum_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: њ
7gradient_tape/DenseLayer/einsum_2/BroadcastGradientArgsBroadcastGradientArgs8gradient_tape/DenseLayer/einsum_2/strided_slice:output:0:gradient_tape/DenseLayer/einsum_2/strided_slice_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџi
'gradient_tape/DenseLayer/einsum_2/add/xConst*
_output_shapes
: *
dtype0*
value	B : Ь
%gradient_tape/DenseLayer/einsum_2/addAddV20gradient_tape/DenseLayer/einsum_2/add/x:output:0<gradient_tape/DenseLayer/einsum_2/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ­
%gradient_tape/DenseLayer/einsum_2/SumSum1gradient_tape/DenseLayer/einsum_2/Einsum:output:0)gradient_tape/DenseLayer/einsum_2/add:z:0*
T0*
_output_shapes
:Ш
)gradient_tape/DenseLayer/einsum_2/ReshapeReshape.gradient_tape/DenseLayer/einsum_2/Sum:output:00gradient_tape/DenseLayer/einsum_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@k
)gradient_tape/DenseLayer/einsum_2/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : а
'gradient_tape/DenseLayer/einsum_2/add_1AddV22gradient_tape/DenseLayer/einsum_2/add_1/x:output:0<gradient_tape/DenseLayer/einsum_2/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџГ
'gradient_tape/DenseLayer/einsum_2/Sum_1Sum3gradient_tape/DenseLayer/einsum_2/Einsum_1:output:0+gradient_tape/DenseLayer/einsum_2/add_1:z:0*
T0*
_output_shapes
:Х
+gradient_tape/DenseLayer/einsum_2/Reshape_1Reshape0gradient_tape/DenseLayer/einsum_2/Sum_1:output:02gradient_tape/DenseLayer/einsum_2/Shape_1:output:0*
T0*
_output_shapes

:@@П
AddN_2AddN'gradient_tape/Y/mul_13/Reshape:output:0'gradient_tape/Y/mul_15/Reshape:output:0'gradient_tape/Y/mul_19/Reshape:output:0)gradient_tape/Y/mul_20/Reshape_1:output:0'gradient_tape/Y/mul_17/Reshape:output:0)gradient_tape/Y/mul_17/Reshape_1:output:0*
N*
T0*#
_output_shapes
:џџџџџџџџџ}
#gradient_tape/Y/strided_slice/ShapeShapeScaledBondVector/truediv:z:0*
T0*
_output_shapes
::эЯ
4gradient_tape/Y/strided_slice/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB"        
2gradient_tape/Y/strided_slice/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB"       
6gradient_tape/Y/strided_slice/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB"      Д
.gradient_tape/Y/strided_slice/StridedSliceGradStridedSliceGrad,gradient_tape/Y/strided_slice/Shape:output:0=gradient_tape/Y/strided_slice/StridedSliceGrad/begin:output:0;gradient_tape/Y/strided_slice/StridedSliceGrad/end:output:0?gradient_tape/Y/strided_slice/StridedSliceGrad/strides:output:0AddN_2:sum:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskП
AddN_3AddN'gradient_tape/Y/mul_14/Reshape:output:0'gradient_tape/Y/mul_16/Reshape:output:0)gradient_tape/Y/mul_19/Reshape_1:output:0'gradient_tape/Y/mul_20/Reshape:output:0'gradient_tape/Y/mul_18/Reshape:output:0)gradient_tape/Y/mul_18/Reshape_1:output:0*
N*
T0*#
_output_shapes
:џџџџџџџџџ
%gradient_tape/Y/strided_slice_1/ShapeShapeScaledBondVector/truediv:z:0*
T0*
_output_shapes
::эЯ
6gradient_tape/Y/strided_slice_1/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB"       
4gradient_tape/Y/strided_slice_1/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB"       
8gradient_tape/Y/strided_slice_1/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB"      О
0gradient_tape/Y/strided_slice_1/StridedSliceGradStridedSliceGrad.gradient_tape/Y/strided_slice_1/Shape:output:0?gradient_tape/Y/strided_slice_1/StridedSliceGrad/begin:output:0=gradient_tape/Y/strided_slice_1/StridedSliceGrad/end:output:0Agradient_tape/Y/strided_slice_1/StridedSliceGrad/strides:output:0AddN_3:sum:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask
gradient_tape/Y/mul_9/MulMulY/strided_slice_3:output:0&gradient_tape/Y/stack/unstack:output:3*
T0*#
_output_shapes
:џџџџџџџџџ^
gradient_tape/Y/mul_9/ShapeConst*
_output_shapes
: *
dtype0*
valueB f
gradient_tape/Y/mul_9/Shape_1ShapeY/add_1:z:0*
T0*
_output_shapes
::эЯ
gradient_tape/Y/mul_11/MulMul&gradient_tape/Y/stack/unstack:output:4	Y/sub:z:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_11/Mul_1MulY/mul_10:z:0&gradient_tape/Y/stack/unstack:output:4*
T0*#
_output_shapes
:џџџџџџџџџf
gradient_tape/Y/mul_11/ShapeShapeY/mul_10:z:0*
T0*
_output_shapes
::эЯe
gradient_tape/Y/mul_11/Shape_1Shape	Y/sub:z:0*
T0*
_output_shapes
::эЯЩ
,gradient_tape/Y/mul_11/BroadcastGradientArgsBroadcastGradientArgs%gradient_tape/Y/mul_11/Shape:output:0'gradient_tape/Y/mul_11/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџГ
gradient_tape/Y/mul_11/SumSumgradient_tape/Y/mul_11/Mul:z:01gradient_tape/Y/mul_11/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Ѓ
gradient_tape/Y/mul_11/ReshapeReshape#gradient_tape/Y/mul_11/Sum:output:0%gradient_tape/Y/mul_11/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџЗ
gradient_tape/Y/mul_11/Sum_1Sum gradient_tape/Y/mul_11/Mul_1:z:01gradient_tape/Y/mul_11/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(Љ
 gradient_tape/Y/mul_11/Reshape_1Reshape%gradient_tape/Y/mul_11/Sum_1:output:0'gradient_tape/Y/mul_11/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_12/MulMulY/strided_slice_5:output:0&gradient_tape/Y/stack/unstack:output:5*
T0*#
_output_shapes
:џџџџџџџџџ_
gradient_tape/Y/mul_12/ShapeConst*
_output_shapes
: *
dtype0*
valueB e
gradient_tape/Y/mul_12/Shape_1Shape	Y/sub:z:0*
T0*
_output_shapes
::эЯд
"gradient_tape/DenseLayer/mul_7/MulMul2gradient_tape/DenseLayer/einsum_2/Reshape:output:0DenseLayer/mul_7/y:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ@
"gradient_tape/DenseLayer/mul_8/MulMul4gradient_tape/DenseLayer/einsum_2/Reshape_1:output:0denselayer_mul_8_y*
T0*
_output_shapes

:@@d
gradient_tape/Y/add_1/ShapeShapeY/mul_7:z:0*
T0*
_output_shapes
::эЯf
gradient_tape/Y/add_1/Shape_1ShapeY/mul_8:z:0*
T0*
_output_shapes
::эЯЦ
+gradient_tape/Y/add_1/BroadcastGradientArgsBroadcastGradientArgs$gradient_tape/Y/add_1/Shape:output:0&gradient_tape/Y/add_1/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџА
gradient_tape/Y/add_1/SumSumgradient_tape/Y/mul_9/Mul:z:00gradient_tape/Y/add_1/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
gradient_tape/Y/add_1/ReshapeReshape"gradient_tape/Y/add_1/Sum:output:0$gradient_tape/Y/add_1/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџВ
gradient_tape/Y/add_1/Sum_1Sumgradient_tape/Y/mul_9/Mul:z:00gradient_tape/Y/add_1/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(І
gradient_tape/Y/add_1/Reshape_1Reshape$gradient_tape/Y/add_1/Sum_1:output:0&gradient_tape/Y/add_1/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_10/MulMul'gradient_tape/Y/mul_11/Reshape:output:0Y/Sqrt_3:y:0*
T0*#
_output_shapes
:џџџџџџџџџИ
AddN_4AddN&gradient_tape/Y/stack/unstack:output:2)gradient_tape/Y/mul_11/Reshape_1:output:0gradient_tape/Y/mul_12/Mul:z:0*
N*
T0*#
_output_shapes
:џџџџџџџџџl
zeros_like_2	ZerosLikeDenseLayer/IdentityN_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ@P
zeros_2Const*
_output_shapes
: *
dtype0*
valueB 2         
mul_19MulDenseLayer/Cast_1:y:0#DenseLayer/einsum_1/Einsum:output:0#^gradient_tape/DenseLayer/mul_7/Mul*
T0*'
_output_shapes
:џџџџџџџџџ@R
	Sigmoid_2Sigmoid
mul_19:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@{
mul_20MulDenseLayer/Cast_1:y:0#DenseLayer/einsum_1/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_4/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?_
sub_4Subsub_4/x:output:0Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_21Mul
mul_20:z:0	sub_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
add_6/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?^
add_6AddV2add_6/x:output:0
mul_21:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_22MulSigmoid_2:y:0	add_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@i
Square_2Square#DenseLayer/einsum_1/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@u
mul_23Mul&gradient_tape/DenseLayer/mul_7/Mul:z:0Square_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_24Mul
mul_23:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_5/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?_
sub_5Subsub_5/x:output:0Sigmoid_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_25Mul
mul_24:z:0	sub_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@X
Const_3Const*
_output_shapes
:*
dtype0*
valueB"       K
Sum_2Sum
mul_25:z:0Const_3:output:0*
T0*
_output_shapes
: s
mul_26Mul&gradient_tape/DenseLayer/mul_7/Mul:z:0
mul_22:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
gradient_tape/Y/mul_7/MulMul&gradient_tape/Y/add_1/Reshape:output:0Y/mul_5:z:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_7/Mul_1MulY/strided_slice_2:output:0&gradient_tape/Y/add_1/Reshape:output:0*
T0*#
_output_shapes
:џџџџџџџџџs
gradient_tape/Y/mul_7/ShapeShapeY/strided_slice_2:output:0*
T0*
_output_shapes
::эЯf
gradient_tape/Y/mul_7/Shape_1ShapeY/mul_5:z:0*
T0*
_output_shapes
::эЯЦ
+gradient_tape/Y/mul_7/BroadcastGradientArgsBroadcastGradientArgs$gradient_tape/Y/mul_7/Shape:output:0&gradient_tape/Y/mul_7/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџА
gradient_tape/Y/mul_7/SumSumgradient_tape/Y/mul_7/Mul:z:00gradient_tape/Y/mul_7/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
gradient_tape/Y/mul_7/ReshapeReshape"gradient_tape/Y/mul_7/Sum:output:0$gradient_tape/Y/mul_7/Shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџД
gradient_tape/Y/mul_7/Sum_1Sumgradient_tape/Y/mul_7/Mul_1:z:00gradient_tape/Y/mul_7/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims(І
gradient_tape/Y/mul_7/Reshape_1Reshape$gradient_tape/Y/mul_7/Sum_1:output:0&gradient_tape/Y/mul_7/Shape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/Y/mul_8/MulMulY/strided_slice_4:output:0(gradient_tape/Y/add_1/Reshape_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
gradient_tape/Y/mul_8/ShapeConst*
_output_shapes
: *
dtype0*
valueB d
gradient_tape/Y/mul_8/Shape_1Shape	Y/add:z:0*
T0*
_output_shapes
::эЯp
gradient_tape/Y/mul_6/MulMulAddN_4:sum:0Y/mul_6/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
AddN_5AddN&gradient_tape/Y/stack/unstack:output:1(gradient_tape/Y/mul_7/Reshape_1:output:0*
N*
T0*#
_output_shapes
:џџџџџџџџџj
gradient_tape/Y/mul_5/MulMulY/Sqrt_1:y:0AddN_5:sum:0*
T0*#
_output_shapes
:џџџџџџџџџ^
gradient_tape/Y/mul_5/ShapeConst*
_output_shapes
: *
dtype0*
valueB u
gradient_tape/Y/mul_5/Shape_1ShapeY/strided_slice_2:output:0*
T0*
_output_shapes
::эЯ
AddN_6AddN&gradient_tape/Y/stack/unstack:output:0gradient_tape/Y/mul_8/Mul:z:0*
N*
T0*#
_output_shapes
:џџџџџџџџџp
gradient_tape/Y/mul_2/MulMulAddN_6:sum:0Y/mul_2/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџъ
AddN_7AddNgradient_tape/Y/mul_10/Mul:z:0&gradient_tape/Y/mul_7/Reshape:output:0gradient_tape/Y/mul_6/Mul:z:0gradient_tape/Y/mul_5/Mul:z:0gradient_tape/Y/mul_2/Mul:z:0*
N*
T0*#
_output_shapes
:џџџџџџџџџ
%gradient_tape/Y/strided_slice_2/ShapeShapeScaledBondVector/truediv:z:0*
T0*
_output_shapes
::эЯ
6gradient_tape/Y/strided_slice_2/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB"       
4gradient_tape/Y/strided_slice_2/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB"       
8gradient_tape/Y/strided_slice_2/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB"      О
0gradient_tape/Y/strided_slice_2/StridedSliceGradStridedSliceGrad.gradient_tape/Y/strided_slice_2/Shape:output:0?gradient_tape/Y/strided_slice_2/StridedSliceGrad/begin:output:0=gradient_tape/Y/strided_slice_2/StridedSliceGrad/end:output:0Agradient_tape/Y/strided_slice_2/StridedSliceGrad/strides:output:0AddN_7:sum:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_masky
'gradient_tape/DenseLayer/einsum_1/ShapeShapeDenseLayer/mul_3:z:0*
T0*
_output_shapes
::эЯz
)gradient_tape/DenseLayer/einsum_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"@   @   Џ
(gradient_tape/DenseLayer/einsum_1/EinsumEinsum
mul_26:z:0DenseLayer/mul_4:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@*
equation...n,...kn->...kЕ
*gradient_tape/DenseLayer/einsum_1/Einsum_1Einsum
mul_26:z:0DenseLayer/mul_3:z:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ@@*
equation...n,...k->...kn
5gradient_tape/DenseLayer/einsum_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7gradient_tape/DenseLayer/einsum_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
7gradient_tape/DenseLayer/einsum_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
/gradient_tape/DenseLayer/einsum_1/strided_sliceStridedSlice0gradient_tape/DenseLayer/einsum_1/Shape:output:0>gradient_tape/DenseLayer/einsum_1/strided_slice/stack:output:0@gradient_tape/DenseLayer/einsum_1/strided_slice/stack_1:output:0@gradient_tape/DenseLayer/einsum_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:
7gradient_tape/DenseLayer/einsum_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9gradient_tape/DenseLayer/einsum_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ
9gradient_tape/DenseLayer/einsum_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
1gradient_tape/DenseLayer/einsum_1/strided_slice_1StridedSlice2gradient_tape/DenseLayer/einsum_1/Shape_1:output:0@gradient_tape/DenseLayer/einsum_1/strided_slice_1/stack:output:0Bgradient_tape/DenseLayer/einsum_1/strided_slice_1/stack_1:output:0Bgradient_tape/DenseLayer/einsum_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: њ
7gradient_tape/DenseLayer/einsum_1/BroadcastGradientArgsBroadcastGradientArgs8gradient_tape/DenseLayer/einsum_1/strided_slice:output:0:gradient_tape/DenseLayer/einsum_1/strided_slice_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџi
'gradient_tape/DenseLayer/einsum_1/add/xConst*
_output_shapes
: *
dtype0*
value	B : Ь
%gradient_tape/DenseLayer/einsum_1/addAddV20gradient_tape/DenseLayer/einsum_1/add/x:output:0<gradient_tape/DenseLayer/einsum_1/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџ­
%gradient_tape/DenseLayer/einsum_1/SumSum1gradient_tape/DenseLayer/einsum_1/Einsum:output:0)gradient_tape/DenseLayer/einsum_1/add:z:0*
T0*
_output_shapes
:Ш
)gradient_tape/DenseLayer/einsum_1/ReshapeReshape.gradient_tape/DenseLayer/einsum_1/Sum:output:00gradient_tape/DenseLayer/einsum_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@k
)gradient_tape/DenseLayer/einsum_1/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : а
'gradient_tape/DenseLayer/einsum_1/add_1AddV22gradient_tape/DenseLayer/einsum_1/add_1/x:output:0<gradient_tape/DenseLayer/einsum_1/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџГ
'gradient_tape/DenseLayer/einsum_1/Sum_1Sum3gradient_tape/DenseLayer/einsum_1/Einsum_1:output:0+gradient_tape/DenseLayer/einsum_1/add_1:z:0*
T0*
_output_shapes
:Х
+gradient_tape/DenseLayer/einsum_1/Reshape_1Reshape0gradient_tape/DenseLayer/einsum_1/Sum_1:output:02gradient_tape/DenseLayer/einsum_1/Shape_1:output:0*
T0*
_output_shapes

:@@ј
AddN_8AddN7gradient_tape/Y/strided_slice/StridedSliceGrad:output:09gradient_tape/Y/strided_slice_1/StridedSliceGrad:output:09gradient_tape/Y/strided_slice_2/StridedSliceGrad:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ
.gradient_tape/ScaledBondVector/truediv/RealDivRealDivAddN_8:sum:0ScaledBondVector/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџp
*gradient_tape/ScaledBondVector/truediv/NegNegbond_vector*
T0*'
_output_shapes
:џџџџџџџџџЗ
0gradient_tape/ScaledBondVector/truediv/RealDiv_1RealDiv.gradient_tape/ScaledBondVector/truediv/Neg:y:0ScaledBondVector/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџН
0gradient_tape/ScaledBondVector/truediv/RealDiv_2RealDiv4gradient_tape/ScaledBondVector/truediv/RealDiv_1:z:0ScaledBondVector/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџЇ
*gradient_tape/ScaledBondVector/truediv/mulMulAddN_8:sum:04gradient_tape/ScaledBondVector/truediv/RealDiv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџu
,gradient_tape/ScaledBondVector/truediv/ShapeShapebond_vector*
T0*
_output_shapes
::эЯ
.gradient_tape/ScaledBondVector/truediv/Shape_1ShapeScaledBondVector/add:z:0*
T0*
_output_shapes
::эЯљ
<gradient_tape/ScaledBondVector/truediv/BroadcastGradientArgsBroadcastGradientArgs5gradient_tape/ScaledBondVector/truediv/Shape:output:07gradient_tape/ScaledBondVector/truediv/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџє
*gradient_tape/ScaledBondVector/truediv/SumSum2gradient_tape/ScaledBondVector/truediv/RealDiv:z:0Agradient_tape/ScaledBondVector/truediv/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(з
.gradient_tape/ScaledBondVector/truediv/ReshapeReshape3gradient_tape/ScaledBondVector/truediv/Sum:output:05gradient_tape/ScaledBondVector/truediv/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџђ
,gradient_tape/ScaledBondVector/truediv/Sum_1Sum.gradient_tape/ScaledBondVector/truediv/mul:z:0Agradient_tape/ScaledBondVector/truediv/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(н
0gradient_tape/ScaledBondVector/truediv/Reshape_1Reshape5gradient_tape/ScaledBondVector/truediv/Sum_1:output:07gradient_tape/ScaledBondVector/truediv/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџд
"gradient_tape/DenseLayer/mul_3/MulMul2gradient_tape/DenseLayer/einsum_1/Reshape:output:0DenseLayer/mul_3/y:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ@
"gradient_tape/DenseLayer/mul_4/MulMul4gradient_tape/DenseLayer/einsum_1/Reshape_1:output:0denselayer_mul_4_y*
T0*
_output_shapes

:@@j
zeros_like_3	ZerosLikeDenseLayer/IdentityN:output:1*
T0*'
_output_shapes
:џџџџџџџџџ@P
zeros_3Const*
_output_shapes
: *
dtype0*
valueB 2        
mul_27MulDenseLayer/Cast:y:0!DenseLayer/einsum/Einsum:output:0#^gradient_tape/DenseLayer/mul_3/Mul*
T0*'
_output_shapes
:џџџџџџџџџ@R
	Sigmoid_3Sigmoid
mul_27:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
mul_28MulDenseLayer/Cast:y:0!DenseLayer/einsum/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_6/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?_
sub_6Subsub_6/x:output:0Sigmoid_3:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_29Mul
mul_28:z:0	sub_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
add_7/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?^
add_7AddV2add_7/x:output:0
mul_29:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_30MulSigmoid_3:y:0	add_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@g
Square_3Square!DenseLayer/einsum/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@u
mul_31Mul&gradient_tape/DenseLayer/mul_3/Mul:z:0Square_3:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_32Mul
mul_31:z:0Sigmoid_3:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_7/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?_
sub_7Subsub_7/x:output:0Sigmoid_3:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_33Mul
mul_32:z:0	sub_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@X
Const_4Const*
_output_shapes
:*
dtype0*
valueB"       K
Sum_3Sum
mul_33:z:0Const_4:output:0*
T0*
_output_shapes
: s
mul_34Mul&gradient_tape/DenseLayer/mul_3/Mul:z:0
mul_30:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
%gradient_tape/DenseLayer/einsum/ShapeShape1BondSpecificRadialBasisFunction/SelectV2:output:0*
T0*
_output_shapes
::эЯx
'gradient_tape/DenseLayer/einsum/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   @   Ћ
&gradient_tape/DenseLayer/einsum/EinsumEinsum
mul_34:z:0DenseLayer/mul:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ*
equation...n,...kn->...kа
(gradient_tape/DenseLayer/einsum/Einsum_1Einsum
mul_34:z:01BondSpecificRadialBasisFunction/SelectV2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ@*
equation...n,...k->...kn}
3gradient_tape/DenseLayer/einsum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5gradient_tape/DenseLayer/einsum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
5gradient_tape/DenseLayer/einsum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
-gradient_tape/DenseLayer/einsum/strided_sliceStridedSlice.gradient_tape/DenseLayer/einsum/Shape:output:0<gradient_tape/DenseLayer/einsum/strided_slice/stack:output:0>gradient_tape/DenseLayer/einsum/strided_slice/stack_1:output:0>gradient_tape/DenseLayer/einsum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:
5gradient_tape/DenseLayer/einsum/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7gradient_tape/DenseLayer/einsum/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ
7gradient_tape/DenseLayer/einsum/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
/gradient_tape/DenseLayer/einsum/strided_slice_1StridedSlice0gradient_tape/DenseLayer/einsum/Shape_1:output:0>gradient_tape/DenseLayer/einsum/strided_slice_1/stack:output:0@gradient_tape/DenseLayer/einsum/strided_slice_1/stack_1:output:0@gradient_tape/DenseLayer/einsum/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: є
5gradient_tape/DenseLayer/einsum/BroadcastGradientArgsBroadcastGradientArgs6gradient_tape/DenseLayer/einsum/strided_slice:output:08gradient_tape/DenseLayer/einsum/strided_slice_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџg
%gradient_tape/DenseLayer/einsum/add/xConst*
_output_shapes
: *
dtype0*
value	B : Ц
#gradient_tape/DenseLayer/einsum/addAddV2.gradient_tape/DenseLayer/einsum/add/x:output:0:gradient_tape/DenseLayer/einsum/BroadcastGradientArgs:r0:0*
T0*#
_output_shapes
:џџџџџџџџџЇ
#gradient_tape/DenseLayer/einsum/SumSum/gradient_tape/DenseLayer/einsum/Einsum:output:0'gradient_tape/DenseLayer/einsum/add:z:0*
T0*
_output_shapes
:Т
'gradient_tape/DenseLayer/einsum/ReshapeReshape,gradient_tape/DenseLayer/einsum/Sum:output:0.gradient_tape/DenseLayer/einsum/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџi
'gradient_tape/DenseLayer/einsum/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : Ъ
%gradient_tape/DenseLayer/einsum/add_1AddV20gradient_tape/DenseLayer/einsum/add_1/x:output:0:gradient_tape/DenseLayer/einsum/BroadcastGradientArgs:r1:0*
T0*#
_output_shapes
:џџџџџџџџџ­
%gradient_tape/DenseLayer/einsum/Sum_1Sum1gradient_tape/DenseLayer/einsum/Einsum_1:output:0)gradient_tape/DenseLayer/einsum/add_1:z:0*
T0*
_output_shapes
:П
)gradient_tape/DenseLayer/einsum/Reshape_1Reshape.gradient_tape/DenseLayer/einsum/Sum_1:output:00gradient_tape/DenseLayer/einsum/Shape_1:output:0*
T0*
_output_shapes

:@|
3gradient_tape/BondSpecificRadialBasisFunction/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        
6gradient_tape/BondSpecificRadialBasisFunction/SelectV2SelectV20BondSpecificRadialBasisFunction/GreaterEqual:z:00gradient_tape/DenseLayer/einsum/Reshape:output:0<gradient_tape/BondSpecificRadialBasisFunction/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
8gradient_tape/BondSpecificRadialBasisFunction/SelectV2_1SelectV20BondSpecificRadialBasisFunction/GreaterEqual:z:0<gradient_tape/BondSpecificRadialBasisFunction/zeros:output:00gradient_tape/DenseLayer/einsum/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
3gradient_tape/BondSpecificRadialBasisFunction/ShapeShape.BondSpecificRadialBasisFunction/zeros_like:y:0*
T0*
_output_shapes
::эЯЄ
5gradient_tape/BondSpecificRadialBasisFunction/Shape_1Shape1BondSpecificRadialBasisFunction/SelectV2:output:0*
T0*
_output_shapes
::эЯ
Cgradient_tape/BondSpecificRadialBasisFunction/BroadcastGradientArgsBroadcastGradientArgs<gradient_tape/BondSpecificRadialBasisFunction/Shape:output:0>gradient_tape/BondSpecificRadialBasisFunction/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
1gradient_tape/BondSpecificRadialBasisFunction/SumSum?gradient_tape/BondSpecificRadialBasisFunction/SelectV2:output:0Hgradient_tape/BondSpecificRadialBasisFunction/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ь
5gradient_tape/BondSpecificRadialBasisFunction/ReshapeReshape:gradient_tape/BondSpecificRadialBasisFunction/Sum:output:0<gradient_tape/BondSpecificRadialBasisFunction/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
5gradient_tape/BondSpecificRadialBasisFunction/Shape_2Shape)BondSpecificRadialBasisFunction/mul_7:z:0*
T0*
_output_shapes
::эЯЄ
5gradient_tape/BondSpecificRadialBasisFunction/Shape_3Shape1BondSpecificRadialBasisFunction/SelectV2:output:0*
T0*
_output_shapes
::эЯ
Egradient_tape/BondSpecificRadialBasisFunction/BroadcastGradientArgs_1BroadcastGradientArgs>gradient_tape/BondSpecificRadialBasisFunction/Shape_2:output:0>gradient_tape/BondSpecificRadialBasisFunction/Shape_3:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
3gradient_tape/BondSpecificRadialBasisFunction/Sum_1SumAgradient_tape/BondSpecificRadialBasisFunction/SelectV2_1:output:0Jgradient_tape/BondSpecificRadialBasisFunction/BroadcastGradientArgs_1:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ђ
7gradient_tape/BondSpecificRadialBasisFunction/Reshape_1Reshape<gradient_tape/BondSpecificRadialBasisFunction/Sum_1:output:0>gradient_tape/BondSpecificRadialBasisFunction/Shape_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
 gradient_tape/DenseLayer/mul/MulMul2gradient_tape/DenseLayer/einsum/Reshape_1:output:0denselayer_mul_y*
T0*
_output_shapes

:@н
7gradient_tape/BondSpecificRadialBasisFunction/mul_7/MulMul@gradient_tape/BondSpecificRadialBasisFunction/Reshape_1:output:0)BondSpecificRadialBasisFunction/sub_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџю
9gradient_tape/BondSpecificRadialBasisFunction/mul_7/Mul_1Mul8BondSpecificRadialBasisFunction/strided_slice_1:output:0@gradient_tape/BondSpecificRadialBasisFunction/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџЏ
9gradient_tape/BondSpecificRadialBasisFunction/mul_7/ShapeShape8BondSpecificRadialBasisFunction/strided_slice_1:output:0*
T0*
_output_shapes
::эЯЂ
;gradient_tape/BondSpecificRadialBasisFunction/mul_7/Shape_1Shape)BondSpecificRadialBasisFunction/sub_5:z:0*
T0*
_output_shapes
::эЯ 
Igradient_tape/BondSpecificRadialBasisFunction/mul_7/BroadcastGradientArgsBroadcastGradientArgsBgradient_tape/BondSpecificRadialBasisFunction/mul_7/Shape:output:0Dgradient_tape/BondSpecificRadialBasisFunction/mul_7/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
7gradient_tape/BondSpecificRadialBasisFunction/mul_7/SumSum;gradient_tape/BondSpecificRadialBasisFunction/mul_7/Mul:z:0Ngradient_tape/BondSpecificRadialBasisFunction/mul_7/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ў
;gradient_tape/BondSpecificRadialBasisFunction/mul_7/ReshapeReshape@gradient_tape/BondSpecificRadialBasisFunction/mul_7/Sum:output:0Bgradient_tape/BondSpecificRadialBasisFunction/mul_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
9gradient_tape/BondSpecificRadialBasisFunction/mul_7/Sum_1Sum=gradient_tape/BondSpecificRadialBasisFunction/mul_7/Mul_1:z:0Ngradient_tape/BondSpecificRadialBasisFunction/mul_7/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
=gradient_tape/BondSpecificRadialBasisFunction/mul_7/Reshape_1ReshapeBgradient_tape/BondSpecificRadialBasisFunction/mul_7/Sum_1:output:0Dgradient_tape/BondSpecificRadialBasisFunction/mul_7/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџЗ
Cgradient_tape/BondSpecificRadialBasisFunction/strided_slice_1/ShapeShape6BondSpecificRadialBasisFunction/strided_slice:output:0*
T0*
_output_shapes
::эЯЅ
Tgradient_tape/BondSpecificRadialBasisFunction/strided_slice_1/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB"       Ѓ
Rgradient_tape/BondSpecificRadialBasisFunction/strided_slice_1/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB"        Ї
Vgradient_tape/BondSpecificRadialBasisFunction/strided_slice_1/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB"      є
Ngradient_tape/BondSpecificRadialBasisFunction/strided_slice_1/StridedSliceGradStridedSliceGradLgradient_tape/BondSpecificRadialBasisFunction/strided_slice_1/Shape:output:0]gradient_tape/BondSpecificRadialBasisFunction/strided_slice_1/StridedSliceGrad/begin:output:0[gradient_tape/BondSpecificRadialBasisFunction/strided_slice_1/StridedSliceGrad/end:output:0_gradient_tape/BondSpecificRadialBasisFunction/strided_slice_1/StridedSliceGrad/strides:output:0Dgradient_tape/BondSpecificRadialBasisFunction/mul_7/Reshape:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskИ
7gradient_tape/BondSpecificRadialBasisFunction/sub_5/NegNegFgradient_tape/BondSpecificRadialBasisFunction/mul_7/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
9gradient_tape/BondSpecificRadialBasisFunction/sub_5/ShapeShape)BondSpecificRadialBasisFunction/add_1:z:0*
T0*
_output_shapes
::эЯЂ
;gradient_tape/BondSpecificRadialBasisFunction/sub_5/Shape_1Shape)BondSpecificRadialBasisFunction/mul_6:z:0*
T0*
_output_shapes
::эЯ 
Igradient_tape/BondSpecificRadialBasisFunction/sub_5/BroadcastGradientArgsBroadcastGradientArgsBgradient_tape/BondSpecificRadialBasisFunction/sub_5/Shape:output:0Dgradient_tape/BondSpecificRadialBasisFunction/sub_5/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЂ
7gradient_tape/BondSpecificRadialBasisFunction/sub_5/SumSumFgradient_tape/BondSpecificRadialBasisFunction/mul_7/Reshape_1:output:0Ngradient_tape/BondSpecificRadialBasisFunction/sub_5/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ў
;gradient_tape/BondSpecificRadialBasisFunction/sub_5/ReshapeReshape@gradient_tape/BondSpecificRadialBasisFunction/sub_5/Sum:output:0Bgradient_tape/BondSpecificRadialBasisFunction/sub_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
9gradient_tape/BondSpecificRadialBasisFunction/sub_5/Sum_1Sum;gradient_tape/BondSpecificRadialBasisFunction/sub_5/Neg:y:0Ngradient_tape/BondSpecificRadialBasisFunction/sub_5/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
=gradient_tape/BondSpecificRadialBasisFunction/sub_5/Reshape_1ReshapeBgradient_tape/BondSpecificRadialBasisFunction/sub_5/Sum_1:output:0Dgradient_tape/BondSpecificRadialBasisFunction/sub_5/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
Agradient_tape/BondSpecificRadialBasisFunction/strided_slice/ShapeShape-BondSpecificRadialBasisFunction/transpose:y:0*
T0*
_output_shapes
::эЯЇ
Rgradient_tape/BondSpecificRadialBasisFunction/strided_slice/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*!
valueB"            Ѕ
Pgradient_tape/BondSpecificRadialBasisFunction/strided_slice/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*!
valueB"           Љ
Tgradient_tape/BondSpecificRadialBasisFunction/strided_slice/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*!
valueB"         
Lgradient_tape/BondSpecificRadialBasisFunction/strided_slice/StridedSliceGradStridedSliceGradJgradient_tape/BondSpecificRadialBasisFunction/strided_slice/Shape:output:0[gradient_tape/BondSpecificRadialBasisFunction/strided_slice/StridedSliceGrad/begin:output:0Ygradient_tape/BondSpecificRadialBasisFunction/strided_slice/StridedSliceGrad/end:output:0]gradient_tape/BondSpecificRadialBasisFunction/strided_slice/StridedSliceGrad/strides:output:0Wgradient_tape/BondSpecificRadialBasisFunction/strided_slice_1/StridedSliceGrad:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask 
9gradient_tape/BondSpecificRadialBasisFunction/add_1/ShapeShape)BondSpecificRadialBasisFunction/sub_4:z:0*
T0*
_output_shapes
::эЯЂ
;gradient_tape/BondSpecificRadialBasisFunction/add_1/Shape_1Shape)BondSpecificRadialBasisFunction/mul_5:z:0*
T0*
_output_shapes
::эЯ 
Igradient_tape/BondSpecificRadialBasisFunction/add_1/BroadcastGradientArgsBroadcastGradientArgsBgradient_tape/BondSpecificRadialBasisFunction/add_1/Shape:output:0Dgradient_tape/BondSpecificRadialBasisFunction/add_1/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ 
7gradient_tape/BondSpecificRadialBasisFunction/add_1/SumSumDgradient_tape/BondSpecificRadialBasisFunction/sub_5/Reshape:output:0Ngradient_tape/BondSpecificRadialBasisFunction/add_1/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ў
;gradient_tape/BondSpecificRadialBasisFunction/add_1/ReshapeReshape@gradient_tape/BondSpecificRadialBasisFunction/add_1/Sum:output:0Bgradient_tape/BondSpecificRadialBasisFunction/add_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
9gradient_tape/BondSpecificRadialBasisFunction/add_1/Sum_1SumDgradient_tape/BondSpecificRadialBasisFunction/sub_5/Reshape:output:0Ngradient_tape/BondSpecificRadialBasisFunction/add_1/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
=gradient_tape/BondSpecificRadialBasisFunction/add_1/Reshape_1ReshapeBgradient_tape/BondSpecificRadialBasisFunction/add_1/Sum_1:output:0Dgradient_tape/BondSpecificRadialBasisFunction/add_1/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџъ
7gradient_tape/BondSpecificRadialBasisFunction/mul_6/MulMul0BondSpecificRadialBasisFunction/mul_6/x:output:0Fgradient_tape/BondSpecificRadialBasisFunction/sub_5/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ|
9gradient_tape/BondSpecificRadialBasisFunction/mul_6/ShapeConst*
_output_shapes
: *
dtype0*
valueB Ђ
;gradient_tape/BondSpecificRadialBasisFunction/mul_6/Shape_1Shape)BondSpecificRadialBasisFunction/pow_2:z:0*
T0*
_output_shapes
::эЯГ
Igradient_tape/BondSpecificRadialBasisFunction/transpose/InvertPermutationInvertPermutation7BondSpecificRadialBasisFunction/transpose/perm:output:0*
_output_shapes
:Њ
Agradient_tape/BondSpecificRadialBasisFunction/transpose/transpose	TransposeUgradient_tape/BondSpecificRadialBasisFunction/strided_slice/StridedSliceGrad:output:0Mgradient_tape/BondSpecificRadialBasisFunction/transpose/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџЖ
7gradient_tape/BondSpecificRadialBasisFunction/sub_4/NegNegDgradient_tape/BondSpecificRadialBasisFunction/add_1/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ|
9gradient_tape/BondSpecificRadialBasisFunction/sub_4/ShapeConst*
_output_shapes
: *
dtype0*
valueB Ђ
;gradient_tape/BondSpecificRadialBasisFunction/sub_4/Shape_1Shape)BondSpecificRadialBasisFunction/mul_4:z:0*
T0*
_output_shapes
::эЯъ
7gradient_tape/BondSpecificRadialBasisFunction/mul_5/MulMul0BondSpecificRadialBasisFunction/mul_5/x:output:0Fgradient_tape/BondSpecificRadialBasisFunction/add_1/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ|
9gradient_tape/BondSpecificRadialBasisFunction/mul_5/ShapeConst*
_output_shapes
: *
dtype0*
valueB Ђ
;gradient_tape/BondSpecificRadialBasisFunction/mul_5/Shape_1Shape)BondSpecificRadialBasisFunction/pow_1:z:0*
T0*
_output_shapes
::эЯп
7gradient_tape/BondSpecificRadialBasisFunction/pow_2/mulMul;gradient_tape/BondSpecificRadialBasisFunction/mul_6/Mul:z:00BondSpecificRadialBasisFunction/pow_2/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
9gradient_tape/BondSpecificRadialBasisFunction/pow_2/sub/yConst*
_output_shapes
: *
dtype0*
valueB 2      №?е
7gradient_tape/BondSpecificRadialBasisFunction/pow_2/subSub0BondSpecificRadialBasisFunction/pow_2/y:output:0Bgradient_tape/BondSpecificRadialBasisFunction/pow_2/sub/y:output:0*
T0*
_output_shapes
: к
7gradient_tape/BondSpecificRadialBasisFunction/pow_2/PowPow+BondSpecificRadialBasisFunction/truediv:z:0;gradient_tape/BondSpecificRadialBasisFunction/pow_2/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџь
9gradient_tape/BondSpecificRadialBasisFunction/pow_2/mul_1Mul;gradient_tape/BondSpecificRadialBasisFunction/pow_2/mul:z:0;gradient_tape/BondSpecificRadialBasisFunction/pow_2/Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџя
;gradient_tape/BondSpecificRadialBasisFunction/stack/unstackUnpackEgradient_tape/BondSpecificRadialBasisFunction/transpose/transpose:y:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*	
numп
7gradient_tape/BondSpecificRadialBasisFunction/mul_4/MulMul0BondSpecificRadialBasisFunction/mul_4/x:output:0;gradient_tape/BondSpecificRadialBasisFunction/sub_4/Neg:y:0*
T0*'
_output_shapes
:џџџџџџџџџ|
9gradient_tape/BondSpecificRadialBasisFunction/mul_4/ShapeConst*
_output_shapes
: *
dtype0*
valueB  
;gradient_tape/BondSpecificRadialBasisFunction/mul_4/Shape_1Shape'BondSpecificRadialBasisFunction/pow:z:0*
T0*
_output_shapes
::эЯп
7gradient_tape/BondSpecificRadialBasisFunction/pow_1/mulMul;gradient_tape/BondSpecificRadialBasisFunction/mul_5/Mul:z:00BondSpecificRadialBasisFunction/pow_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
9gradient_tape/BondSpecificRadialBasisFunction/pow_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB 2      №?е
7gradient_tape/BondSpecificRadialBasisFunction/pow_1/subSub0BondSpecificRadialBasisFunction/pow_1/y:output:0Bgradient_tape/BondSpecificRadialBasisFunction/pow_1/sub/y:output:0*
T0*
_output_shapes
: к
7gradient_tape/BondSpecificRadialBasisFunction/pow_1/PowPow+BondSpecificRadialBasisFunction/truediv:z:0;gradient_tape/BondSpecificRadialBasisFunction/pow_1/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџь
9gradient_tape/BondSpecificRadialBasisFunction/pow_1/mul_1Mul;gradient_tape/BondSpecificRadialBasisFunction/pow_1/mul:z:0;gradient_tape/BondSpecificRadialBasisFunction/pow_1/Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџЖ
7gradient_tape/BondSpecificRadialBasisFunction/sub_3/NegNegDgradient_tape/BondSpecificRadialBasisFunction/stack/unstack:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 
9gradient_tape/BondSpecificRadialBasisFunction/sub_3/ShapeShape)BondSpecificRadialBasisFunction/mul_3:z:0*
T0*
_output_shapes
::эЯІ
;gradient_tape/BondSpecificRadialBasisFunction/sub_3/Shape_1Shape-BondSpecificRadialBasisFunction/ones_like:y:0*
T0*
_output_shapes
::эЯ 
Igradient_tape/BondSpecificRadialBasisFunction/sub_3/BroadcastGradientArgsBroadcastGradientArgsBgradient_tape/BondSpecificRadialBasisFunction/sub_3/Shape:output:0Dgradient_tape/BondSpecificRadialBasisFunction/sub_3/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ 
7gradient_tape/BondSpecificRadialBasisFunction/sub_3/SumSumDgradient_tape/BondSpecificRadialBasisFunction/stack/unstack:output:2Ngradient_tape/BondSpecificRadialBasisFunction/sub_3/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ў
;gradient_tape/BondSpecificRadialBasisFunction/sub_3/ReshapeReshape@gradient_tape/BondSpecificRadialBasisFunction/sub_3/Sum:output:0Bgradient_tape/BondSpecificRadialBasisFunction/sub_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
9gradient_tape/BondSpecificRadialBasisFunction/sub_3/Sum_1Sum;gradient_tape/BondSpecificRadialBasisFunction/sub_3/Neg:y:0Ngradient_tape/BondSpecificRadialBasisFunction/sub_3/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
=gradient_tape/BondSpecificRadialBasisFunction/sub_3/Reshape_1ReshapeBgradient_tape/BondSpecificRadialBasisFunction/sub_3/Sum_1:output:0Dgradient_tape/BondSpecificRadialBasisFunction/sub_3/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџл
5gradient_tape/BondSpecificRadialBasisFunction/pow/mulMul;gradient_tape/BondSpecificRadialBasisFunction/mul_4/Mul:z:0.BondSpecificRadialBasisFunction/pow/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
7gradient_tape/BondSpecificRadialBasisFunction/pow/sub/yConst*
_output_shapes
: *
dtype0*
valueB 2      №?Я
5gradient_tape/BondSpecificRadialBasisFunction/pow/subSub.BondSpecificRadialBasisFunction/pow/y:output:0@gradient_tape/BondSpecificRadialBasisFunction/pow/sub/y:output:0*
T0*
_output_shapes
: ж
5gradient_tape/BondSpecificRadialBasisFunction/pow/PowPow+BondSpecificRadialBasisFunction/truediv:z:09gradient_tape/BondSpecificRadialBasisFunction/pow/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџц
7gradient_tape/BondSpecificRadialBasisFunction/pow/mul_1Mul9gradient_tape/BondSpecificRadialBasisFunction/pow/mul:z:09gradient_tape/BondSpecificRadialBasisFunction/pow/Pow:z:0*
T0*'
_output_shapes
:џџџџџџџџџс
7gradient_tape/BondSpecificRadialBasisFunction/mul_3/MulMulDgradient_tape/BondSpecificRadialBasisFunction/sub_3/Reshape:output:0)BondSpecificRadialBasisFunction/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџу
9gradient_tape/BondSpecificRadialBasisFunction/mul_3/Mul_1Mul)BondSpecificRadialBasisFunction/sub_2:z:0Dgradient_tape/BondSpecificRadialBasisFunction/sub_3/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
9gradient_tape/BondSpecificRadialBasisFunction/mul_3/ShapeShape)BondSpecificRadialBasisFunction/sub_2:z:0*
T0*
_output_shapes
::эЯЂ
;gradient_tape/BondSpecificRadialBasisFunction/mul_3/Shape_1Shape)BondSpecificRadialBasisFunction/mul_2:z:0*
T0*
_output_shapes
::эЯ 
Igradient_tape/BondSpecificRadialBasisFunction/mul_3/BroadcastGradientArgsBroadcastGradientArgsBgradient_tape/BondSpecificRadialBasisFunction/mul_3/Shape:output:0Dgradient_tape/BondSpecificRadialBasisFunction/mul_3/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
7gradient_tape/BondSpecificRadialBasisFunction/mul_3/SumSum;gradient_tape/BondSpecificRadialBasisFunction/mul_3/Mul:z:0Ngradient_tape/BondSpecificRadialBasisFunction/mul_3/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ў
;gradient_tape/BondSpecificRadialBasisFunction/mul_3/ReshapeReshape@gradient_tape/BondSpecificRadialBasisFunction/mul_3/Sum:output:0Bgradient_tape/BondSpecificRadialBasisFunction/mul_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
9gradient_tape/BondSpecificRadialBasisFunction/mul_3/Sum_1Sum=gradient_tape/BondSpecificRadialBasisFunction/mul_3/Mul_1:z:0Ngradient_tape/BondSpecificRadialBasisFunction/mul_3/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
=gradient_tape/BondSpecificRadialBasisFunction/mul_3/Reshape_1ReshapeBgradient_tape/BondSpecificRadialBasisFunction/mul_3/Sum_1:output:0Dgradient_tape/BondSpecificRadialBasisFunction/mul_3/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџз
AddN_9AddNDgradient_tape/BondSpecificRadialBasisFunction/stack/unstack:output:0Fgradient_tape/BondSpecificRadialBasisFunction/sub_3/Reshape_1:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџъ
7gradient_tape/BondSpecificRadialBasisFunction/mul_2/MulMul0BondSpecificRadialBasisFunction/mul_2/x:output:0Fgradient_tape/BondSpecificRadialBasisFunction/mul_3/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ|
9gradient_tape/BondSpecificRadialBasisFunction/mul_2/ShapeConst*
_output_shapes
: *
dtype0*
valueB Ђ
;gradient_tape/BondSpecificRadialBasisFunction/mul_2/Shape_1Shape)BondSpecificRadialBasisFunction/sub_2:z:0*
T0*
_output_shapes
::эЯ
AddN_10AddNDgradient_tape/BondSpecificRadialBasisFunction/stack/unstack:output:1Dgradient_tape/BondSpecificRadialBasisFunction/mul_3/Reshape:output:0;gradient_tape/BondSpecificRadialBasisFunction/mul_2/Mul:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџБ
7gradient_tape/BondSpecificRadialBasisFunction/mul_1/MulMul0BondSpecificRadialBasisFunction/mul_1/x:output:0AddN_10:sum:0*
T0*'
_output_shapes
:џџџџџџџџџ|
9gradient_tape/BondSpecificRadialBasisFunction/mul_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB Ђ
;gradient_tape/BondSpecificRadialBasisFunction/mul_1/Shape_1Shape)BondSpecificRadialBasisFunction/sub_1:z:0*
T0*
_output_shapes
::эЯ­
7gradient_tape/BondSpecificRadialBasisFunction/sub_1/NegNeg;gradient_tape/BondSpecificRadialBasisFunction/mul_1/Mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ|
9gradient_tape/BondSpecificRadialBasisFunction/sub_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB  
;gradient_tape/BondSpecificRadialBasisFunction/sub_1/Shape_1Shape'BondSpecificRadialBasisFunction/Abs:y:0*
T0*
_output_shapes
::эЯ
6gradient_tape/BondSpecificRadialBasisFunction/Abs/SignSign'BondSpecificRadialBasisFunction/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџч
5gradient_tape/BondSpecificRadialBasisFunction/Abs/mulMul;gradient_tape/BondSpecificRadialBasisFunction/sub_1/Neg:y:0:gradient_tape/BondSpecificRadialBasisFunction/Abs/Sign:y:0*
T0*'
_output_shapes
:џџџџџџџџџЉ
5gradient_tape/BondSpecificRadialBasisFunction/sub/NegNeg9gradient_tape/BondSpecificRadialBasisFunction/Abs/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџz
7gradient_tape/BondSpecificRadialBasisFunction/sub/ShapeConst*
_output_shapes
: *
dtype0*
valueB Ђ
9gradient_tape/BondSpecificRadialBasisFunction/sub/Shape_1Shape+BondSpecificRadialBasisFunction/truediv:z:0*
T0*
_output_shapes
::эЯР
AddN_11AddN=gradient_tape/BondSpecificRadialBasisFunction/pow_2/mul_1:z:0=gradient_tape/BondSpecificRadialBasisFunction/pow_1/mul_1:z:0;gradient_tape/BondSpecificRadialBasisFunction/pow/mul_1:z:09gradient_tape/BondSpecificRadialBasisFunction/sub/Neg:y:0*
N*
T0*'
_output_shapes
:џџџџџџџџџК
=gradient_tape/BondSpecificRadialBasisFunction/truediv/RealDivRealDivAddN_11:sum:0/BondSpecificRadialBasisFunction/Gather:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
9gradient_tape/BondSpecificRadialBasisFunction/truediv/NegNegBondLength/norm/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџь
?gradient_tape/BondSpecificRadialBasisFunction/truediv/RealDiv_1RealDiv=gradient_tape/BondSpecificRadialBasisFunction/truediv/Neg:y:0/BondSpecificRadialBasisFunction/Gather:output:0*
T0*'
_output_shapes
:џџџџџџџџџђ
?gradient_tape/BondSpecificRadialBasisFunction/truediv/RealDiv_2RealDivCgradient_tape/BondSpecificRadialBasisFunction/truediv/RealDiv_1:z:0/BondSpecificRadialBasisFunction/Gather:output:0*
T0*'
_output_shapes
:џџџџџџџџџЦ
9gradient_tape/BondSpecificRadialBasisFunction/truediv/mulMulAddN_11:sum:0Cgradient_tape/BondSpecificRadialBasisFunction/truediv/RealDiv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
;gradient_tape/BondSpecificRadialBasisFunction/truediv/ShapeShapeBondLength/norm/Sqrt:y:0*
T0*
_output_shapes
::эЯЊ
=gradient_tape/BondSpecificRadialBasisFunction/truediv/Shape_1Shape/BondSpecificRadialBasisFunction/Gather:output:0*
T0*
_output_shapes
::эЯІ
Kgradient_tape/BondSpecificRadialBasisFunction/truediv/BroadcastGradientArgsBroadcastGradientArgsDgradient_tape/BondSpecificRadialBasisFunction/truediv/Shape:output:0Fgradient_tape/BondSpecificRadialBasisFunction/truediv/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЁ
9gradient_tape/BondSpecificRadialBasisFunction/truediv/SumSumAgradient_tape/BondSpecificRadialBasisFunction/truediv/RealDiv:z:0Pgradient_tape/BondSpecificRadialBasisFunction/truediv/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
=gradient_tape/BondSpecificRadialBasisFunction/truediv/ReshapeReshapeBgradient_tape/BondSpecificRadialBasisFunction/truediv/Sum:output:0Dgradient_tape/BondSpecificRadialBasisFunction/truediv/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
;gradient_tape/BondSpecificRadialBasisFunction/truediv/Sum_1Sum=gradient_tape/BondSpecificRadialBasisFunction/truediv/mul:z:0Pgradient_tape/BondSpecificRadialBasisFunction/truediv/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
?gradient_tape/BondSpecificRadialBasisFunction/truediv/Reshape_1ReshapeDgradient_tape/BondSpecificRadialBasisFunction/truediv/Sum_1:output:0Fgradient_tape/BondSpecificRadialBasisFunction/truediv/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџЭ
AddN_12AddN9gradient_tape/ScaledBondVector/truediv/Reshape_1:output:0Fgradient_tape/BondSpecificRadialBasisFunction/truediv/Reshape:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ
&gradient_tape/BondLength/norm/SqrtGradSqrtGradBondLength/norm/Sqrt:y:0AddN_12:sum:0*
T0*'
_output_shapes
:џџџџџџџџџx
#gradient_tape/BondLength/norm/ShapeShapeBondLength/norm/mul:z:0*
T0*
_output_shapes
::эЯФ
)gradient_tape/BondLength/norm/BroadcastToBroadcastTo*gradient_tape/BondLength/norm/SqrtGrad:z:0,gradient_tape/BondLength/norm/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
%gradient_tape/BondLength/norm/mul/MulMul2gradient_tape/BondLength/norm/BroadcastTo:output:0bond_vector*
T0*'
_output_shapes
:џџџџџџџџџЁ
'gradient_tape/BondLength/norm/mul/Mul_1Mulbond_vector2gradient_tape/BondLength/norm/BroadcastTo:output:0*
T0*'
_output_shapes
:џџџџџџџџџp
'gradient_tape/BondLength/norm/mul/ShapeShapebond_vector*
T0*
_output_shapes
::эЯr
)gradient_tape/BondLength/norm/mul/Shape_1Shapebond_vector*
T0*
_output_shapes
::эЯъ
7gradient_tape/BondLength/norm/mul/BroadcastGradientArgsBroadcastGradientArgs0gradient_tape/BondLength/norm/mul/Shape:output:02gradient_tape/BondLength/norm/mul/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџс
%gradient_tape/BondLength/norm/mul/SumSum)gradient_tape/BondLength/norm/mul/Mul:z:0<gradient_tape/BondLength/norm/mul/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(Ш
)gradient_tape/BondLength/norm/mul/ReshapeReshape.gradient_tape/BondLength/norm/mul/Sum:output:00gradient_tape/BondLength/norm/mul/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџх
'gradient_tape/BondLength/norm/mul/Sum_1Sum+gradient_tape/BondLength/norm/mul/Mul_1:z:0<gradient_tape/BondLength/norm/mul/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(Ю
+gradient_tape/BondLength/norm/mul/Reshape_1Reshape0gradient_tape/BondLength/norm/mul/Sum_1:output:02gradient_tape/BondLength/norm/mul/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџэ
AddN_13AddN7gradient_tape/ScaledBondVector/truediv/Reshape:output:02gradient_tape/BondLength/norm/mul/Reshape:output:04gradient_tape/BondLength/norm/mul/Reshape_1:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџK
NegNegAddN_13:sum:0*
T0*'
_output_shapes
:џџџџџџџџџY
Sum_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : |
Sum_4SumReshape_1:output:0 Sum_4/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(R
Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB T
Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB `
	Reshape_2Reshapebatch_tot_natReshape_2/shape_1:output:0*
T0*
_output_shapes
: 
UnsortedSegmentSumUnsortedSegmentSumNeg:y:0ind_jReshape_2:output:0*
Tindices0*
T0*'
_output_shapes
:џџџџџџџџџ
UnsortedSegmentSum_1UnsortedSegmentSumNeg:y:0ind_iReshape_2:output:0*
Tindices0*
T0*'
_output_shapes
:џџџџџџџџџz
sub_8SubUnsortedSegmentSum:output:0UnsortedSegmentSum_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџU
mul_35MulNeg:y:0bond_vector*
T0*'
_output_shapes
:џџџџџџџџџY
Sum_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : _
Sum_5Sum
mul_35:z:0 Sum_5/reduction_indices:output:0*
T0*
_output_shapes
:b
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџc
	Reshape_3ReshapeSum_5:output:0Reshape_3/shape:output:0*
T0*
_output_shapes
:f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_3StridedSliceNeg:y:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_4StridedSlicebond_vectorstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_masko
mul_36Mulstrided_slice_3:output:0strided_slice_4:output:0*
T0*#
_output_shapes
:џџџџџџџџџY
Sum_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : [
Sum_6Sum
mul_36:z:0 Sum_6/reduction_indices:output:0*
T0*
_output_shapes
: Y
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:c
	Reshape_4ReshapeSum_6:output:0Reshape_4/shape:output:0*
T0*
_output_shapes
:f
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_5StridedSliceNeg:y:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_6StridedSlicebond_vectorstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_masko
mul_37Mulstrided_slice_5:output:0strided_slice_6:output:0*
T0*#
_output_shapes
:џџџџџџџџџY
Sum_7/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : [
Sum_7Sum
mul_37:z:0 Sum_7/reduction_indices:output:0*
T0*
_output_shapes
: Y
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:c
	Reshape_5ReshapeSum_7:output:0Reshape_5/shape:output:0*
T0*
_output_shapes
:f
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_7StridedSliceNeg:y:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_8StridedSlicebond_vectorstrided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_masko
mul_38Mulstrided_slice_7:output:0strided_slice_8:output:0*
T0*#
_output_shapes
:џџџџџџџџџY
Sum_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : [
Sum_8Sum
mul_38:z:0 Sum_8/reduction_indices:output:0*
T0*
_output_shapes
: Y
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB:c
	Reshape_6ReshapeSum_8:output:0Reshape_6/shape:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : І
concatConcatV2Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:a
IdentityIdentityReshape_1:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџV

Identity_1IdentitySum_4:output:0^NoOp*
T0*
_output_shapes

:Z

Identity_2Identity	sub_8:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS

Identity_3Identityconcat:output:0^NoOp*
T0*
_output_shapes
:щ
NoOpNoOp'^BondSpecificRadialBasisFunction/Gather^ChemIndTransf/ReadVariableOp+^ChemIndTransf/einsum/Einsum/ReadVariableOp^DenseLayer/ReadVariableOp^DenseLayer/ReadVariableOp_1^DenseLayer/ReadVariableOp_2^DenseLayer/ReadVariableOp_3^DenseLayer/ReadVariableOp_4^DenseLayer/ReadVariableOp_5^rho/GatherV2_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*Э
_input_shapesЛ
И:џџџџџџџџџ: : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : :	: ::: : : : : : :: :: : : : : : : : *
	_noinline(2P
&BondSpecificRadialBasisFunction/Gather&BondSpecificRadialBasisFunction/Gather2<
ChemIndTransf/ReadVariableOpChemIndTransf/ReadVariableOp2X
*ChemIndTransf/einsum/Einsum/ReadVariableOp*ChemIndTransf/einsum/Einsum/ReadVariableOp2:
DenseLayer/ReadVariableOp_1DenseLayer/ReadVariableOp_12:
DenseLayer/ReadVariableOp_2DenseLayer/ReadVariableOp_22:
DenseLayer/ReadVariableOp_3DenseLayer/ReadVariableOp_32:
DenseLayer/ReadVariableOp_4DenseLayer/ReadVariableOp_42:
DenseLayer/ReadVariableOp_5DenseLayer/ReadVariableOp_526
DenseLayer/ReadVariableOpDenseLayer/ReadVariableOp2>
rho/GatherV2_1/ReadVariableOprho/GatherV2_1/ReadVariableOp:&

_output_shapes
: :%

_output_shapes
: :$

_output_shapes
: :#

_output_shapes
: :("$
"
_user_specified_name
resource:!

_output_shapes
: :( $
"
_user_specified_name
resource:

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: : 

_output_shapes
::($
"
_user_specified_name
resource:

_output_shapes
: :($
"
_user_specified_name
resource:

_output_shapes
: :($
"
_user_specified_name
resource:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:	:

_output_shapes
: :($
"
_user_specified_name
resource:

_output_shapes
: :($
"
_user_specified_name
resource:

_output_shapes
: :($
"
_user_specified_name
resource:

_output_shapes
: :(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:

_output_shapes
: :IE
#
_output_shapes
:џџџџџџџџџ

_user_specified_namemu_j:IE
#
_output_shapes
:џџџџџџџџџ

_user_specified_namemu_i:JF
#
_output_shapes
:џџџџџџџџџ

_user_specified_nameind_j:JF
#
_output_shapes
:џџџџџџџџџ

_user_specified_nameind_i:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_namebond_vector:JF

_output_shapes
: 
,
_user_specified_namebatch_tot_nat_real:EA

_output_shapes
: 
'
_user_specified_namebatch_tot_nat:P L
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameatomic_mu_i
Ц!
э
"__inference_signature_wrapper_9812
atomic_mu_i
batch_tot_nat
batch_tot_nat_real
bond_vector	
ind_i	
ind_j
mu_i
mu_j
unknown
	unknown_0:
	unknown_1:@
	unknown_2
	unknown_3:@@
	unknown_4
	unknown_5:@@
	unknown_6
	unknown_7:@
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14:

unknown_15

unknown_16:

unknown_17$

unknown_18:

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23:@

unknown_24

unknown_25:@

unknown_26

unknown_27

unknown_28

unknown_29
identity

identity_1

identity_2

identity_3ЂStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallatomic_mu_ibatch_tot_natbatch_tot_nat_realbond_vectorind_iind_jmu_imu_junknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_29*2
Tin+
)2'*
Tout
2*
_XlaMustCompile(*
_collective_manager_ids
 *J
_output_shapes8
6:џџџџџџџџџ::џџџџџџџџџ:*,
_read_only_resource_inputs

	
 "*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_compute_9731o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Э
_input_shapesЛ
И:џџџџџџџџџ: : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : :	: ::: : : : : : :: :: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&

_output_shapes
: :%

_output_shapes
: :$

_output_shapes
: :#

_output_shapes
: :$" 

_user_specified_name9794:!

_output_shapes
: :$  

_user_specified_name9790:

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: : 

_output_shapes
::$ 

_user_specified_name9780:

_output_shapes
: :$ 

_user_specified_name9776:

_output_shapes
: :$ 

_user_specified_name9772:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:	:

_output_shapes
: :$ 

_user_specified_name9758:

_output_shapes
: :$ 

_user_specified_name9754:

_output_shapes
: :$ 

_user_specified_name9750:

_output_shapes
: :$
 

_user_specified_name9746:$	 

_user_specified_name9744:

_output_shapes
: :IE
#
_output_shapes
:џџџџџџџџџ

_user_specified_namemu_j:IE
#
_output_shapes
:џџџџџџџџџ

_user_specified_namemu_i:JF
#
_output_shapes
:џџџџџџџџџ

_user_specified_nameind_j:JF
#
_output_shapes
:џџџџџџџџџ

_user_specified_nameind_i:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_namebond_vector:JF

_output_shapes
: 
,
_user_specified_namebatch_tot_nat_real:EA

_output_shapes
: 
'
_user_specified_namebatch_tot_nat:P L
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameatomic_mu_i
а
М
!__inference_internal_grad_fn_9924
result_grads_0
result_grads_1
result_grads_2
mul_denselayer_cast_1"
mul_denselayer_einsum_1_einsum
identity

identity_1
mulMulmul_denselayer_cast_1mul_denselayer_einsum_1_einsum^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@u
mul_1Mulmul_denselayer_cast_1mul_denselayer_einsum_1_einsum*
T0*'
_output_shapes
:џџџџџџџџџ@N
sub/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
add/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@b
SquareSquaremul_denselayer_einsum_1_einsum*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?]
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:c_
'
_output_shapes
:џџџџџџџџџ@
4
_user_specified_nameDenseLayer/einsum_1/Einsum:IE

_output_shapes
: 
+
_user_specified_nameDenseLayer/Cast_1:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
яl
ж
__inference__traced_save_10045
file_prefix3
!read_disablecopyonread_rbf_cutoff::
,read_1_disablecopyonread_element_map_symbols:8
*read_2_disablecopyonread_element_map_index:>
,read_3_disablecopyonread_z_chemicalembedding:A
'read_4_disablecopyonread_rho_reducing_a:S
Aread_5_disablecopyonread_chemindtransf_denselayer_chemindtransf__:V
Dread_6_disablecopyonread_denselayer_denselayer_denselayer_no_decay_3:@V
Dread_7_disablecopyonread_denselayer_denselayer_denselayer_no_decay_2:@@V
Dread_8_disablecopyonread_denselayer_denselayer_denselayer_no_decay_1:@@T
Bread_9_disablecopyonread_denselayer_denselayer_denselayer_no_decay:@P
>read_10_disablecopyonread_denselayer_denselayer_denselayer___1:@N
<read_11_disablecopyonread_denselayer_denselayer_denselayer__:@
savev2_const_21
identity_25ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: s
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_rbf_cutoff"/device:CPU:0*
_output_shapes
 
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_rbf_cutoff^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_1/DisableCopyOnReadDisableCopyOnRead,read_1_disablecopyonread_element_map_symbols"/device:CPU:0*
_output_shapes
 Ј
Read_1/ReadVariableOpReadVariableOp,read_1_disablecopyonread_element_map_symbols^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_2/DisableCopyOnReadDisableCopyOnRead*read_2_disablecopyonread_element_map_index"/device:CPU:0*
_output_shapes
 І
Read_2/ReadVariableOpReadVariableOp*read_2_disablecopyonread_element_map_index^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_3/DisableCopyOnReadDisableCopyOnRead,read_3_disablecopyonread_z_chemicalembedding"/device:CPU:0*
_output_shapes
 Ќ
Read_3/ReadVariableOpReadVariableOp,read_3_disablecopyonread_z_chemicalembedding^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:{
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_rho_reducing_a"/device:CPU:0*
_output_shapes
 Џ
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_rho_reducing_a^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_5/DisableCopyOnReadDisableCopyOnReadAread_5_disablecopyonread_chemindtransf_denselayer_chemindtransf__"/device:CPU:0*
_output_shapes
 С
Read_5/ReadVariableOpReadVariableOpAread_5_disablecopyonread_chemindtransf_denselayer_chemindtransf__^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_6/DisableCopyOnReadDisableCopyOnReadDread_6_disablecopyonread_denselayer_denselayer_denselayer_no_decay_3"/device:CPU:0*
_output_shapes
 Ф
Read_6/ReadVariableOpReadVariableOpDread_6_disablecopyonread_denselayer_denselayer_denselayer_no_decay_3^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_7/DisableCopyOnReadDisableCopyOnReadDread_7_disablecopyonread_denselayer_denselayer_denselayer_no_decay_2"/device:CPU:0*
_output_shapes
 Ф
Read_7/ReadVariableOpReadVariableOpDread_7_disablecopyonread_denselayer_denselayer_denselayer_no_decay_2^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:@@
Read_8/DisableCopyOnReadDisableCopyOnReadDread_8_disablecopyonread_denselayer_denselayer_denselayer_no_decay_1"/device:CPU:0*
_output_shapes
 Ф
Read_8/ReadVariableOpReadVariableOpDread_8_disablecopyonread_denselayer_denselayer_denselayer_no_decay_1^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:@@
Read_9/DisableCopyOnReadDisableCopyOnReadBread_9_disablecopyonread_denselayer_denselayer_denselayer_no_decay"/device:CPU:0*
_output_shapes
 Т
Read_9/ReadVariableOpReadVariableOpBread_9_disablecopyonread_denselayer_denselayer_denselayer_no_decay^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_10/DisableCopyOnReadDisableCopyOnRead>read_10_disablecopyonread_denselayer_denselayer_denselayer___1"/device:CPU:0*
_output_shapes
 Р
Read_10/ReadVariableOpReadVariableOp>read_10_disablecopyonread_denselayer_denselayer_denselayer___1^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_11/DisableCopyOnReadDisableCopyOnRead<read_11_disablecopyonread_denselayer_denselayer_denselayer__"/device:CPU:0*
_output_shapes
 О
Read_11/ReadVariableOpReadVariableOp<read_11_disablecopyonread_denselayer_denselayer_denselayer__^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:@Џ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*и
valueЮBЫB9instructions/2/bond_cutoff_map/.ATTRIBUTES/VARIABLE_VALUEB=instructions/5/element_map_symbols/.ATTRIBUTES/VARIABLE_VALUEB;instructions/5/element_map_index/.ATTRIBUTES/VARIABLE_VALUEB+instructions/5/w/.ATTRIBUTES/VARIABLE_VALUEB4instructions/7/reducing_A/.ATTRIBUTES/VARIABLE_VALUEB9instructions/6/lin_transform/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer0/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer1/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer2/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer3/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/9/mlp/layer0/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/9/mlp/layer1/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ъ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0savev2_const_21"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_24Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_25IdentityIdentity_24:output:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_25Identity_25:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
: : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:@<

_output_shapes
: 
"
_user_specified_name
Const_21:B>
<
_user_specified_name$"DenseLayer/DenseLayer_DenseLayer__:D@
>
_user_specified_name&$DenseLayer/DenseLayer_DenseLayer___1:I
E
C
_user_specified_name+)DenseLayer/DenseLayer_DenseLayer_no_decay:K	G
E
_user_specified_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_1:KG
E
_user_specified_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_2:KG
E
_user_specified_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_3:HD
B
_user_specified_name*(ChemIndTransf/DenseLayer_ChemIndTransf__:.*
(
_user_specified_namerho/reducing_A:3/
-
_user_specified_nameZ/ChemicalEmbedding:1-
+
_user_specified_nameelement_map_index:3/
-
_user_specified_nameelement_map_symbols:*&
$
_user_specified_name
RBF_cutoff:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix8
!__inference_internal_grad_fn_9897CustomGradient-83018
!__inference_internal_grad_fn_9924CustomGradient-83188
!__inference_internal_grad_fn_9951CustomGradient-83358
!__inference_internal_grad_fn_9978CustomGradient-8663"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ш
serving_defaultд
?
atomic_mu_i0
serving_default_atomic_mu_i:0џџџџџџџџџ
@
batch_tot_nat_real*
$serving_default_batch_tot_nat_real:0 
6
batch_tot_nat%
serving_default_batch_tot_nat:0 
C
bond_vector4
serving_default_bond_vector:0џџџџџџџџџ
3
ind_i*
serving_default_ind_i:0џџџџџџџџџ
3
ind_j*
serving_default_ind_j:0џџџџџџџџџ
1
mu_i)
serving_default_mu_i:0џџџџџџџџџ
1
mu_j)
serving_default_mu_j:0џџџџџџџџџA
atomic_energy0
StatefulPartitionedCall:0џџџџџџџџџ7
total_energy'
StatefulPartitionedCall:1;
total_f0
StatefulPartitionedCall:2џџџџџџџџџ-
virial#
StatefulPartitionedCall:3tensorflow/serving/predict:јV
}
instructions
compute_specs
train_specs

slices
compute

signatures"
_generic_user_object
n
0
1
	2

3
4
5
6
7
8
9
10"
trackable_list_wrapper

	ind_i
	ind_j
bond_vector
mu_i
mu_j
batch_tot_nat
atomic_mu_i
batch_tot_nat_real"
trackable_dict_wrapper
Ч
	ind_i
	ind_j
map_atoms_to_structure
n_struct_total
bond_vector
batch_tot_nat
 mu_i
!mu_j
"atomic_mu_i
#batch_tot_nat_real"
trackable_dict_wrapper
 "
trackable_list_wrapper
К
$trace_02
__inference_compute_9731
В
FullArgSpec
args
j
input_data
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *уЂп
мЊи
0
atomic_mu_i!
atomic_mu_iџџџџџџџџџ
1
batch_tot_nat_real
batch_tot_nat_real 
'
batch_tot_nat
batch_tot_nat 
4
bond_vector%"
bond_vectorџџџџџџџџџ
$
ind_i
ind_iџџџџџџџџџ
$
ind_j
ind_jџџџџџџџџџ
"
mu_i
mu_iџџџџџџџџџ
"
mu_j
mu_jџџџџџџџџџ0z$trace_0
,
%serving_default"
signature_map
.
&
_init_args"
_generic_user_object
.
'
_init_args"
_generic_user_object
T
(
_init_args
)cutoff_dict
*bond_cutoff_map"
_generic_user_object
J
+
_init_args
,hidden_layers
-mlp"
_generic_user_object
6
.
_init_args
/sg"
_generic_user_object
e
0
_init_args
1element_map_symbols
2element_map_index
3w"
_generic_user_object
i
4
_init_args


radial
angular
	indicator
5lin_transform"
_generic_user_object

6
_init_args
7instructions

8ls_max
9allowed_l_p
:	collector
;downscale_embeddings
<
reducing_A"
_generic_user_object
.
=
_init_args"
_generic_user_object
b
>
_init_args

target

?origin
@hidden_layers
Amlp"
_generic_user_object
:
B
_init_args

target"
_generic_user_object
+
	Cshape"
trackable_dict_wrapper
+
	Dshape"
trackable_dict_wrapper
+
	Eshape"
trackable_dict_wrapper
+
	Fshape"
trackable_dict_wrapper
+
	Gshape"
trackable_dict_wrapper
+
	Hshape"
trackable_dict_wrapper
+
	Ishape"
trackable_dict_wrapper
+
	Jshape"
trackable_dict_wrapper
+
	Cshape"
trackable_dict_wrapper
+
	Kshape"
trackable_dict_wrapper
+
	Lshape"
trackable_dict_wrapper
+
	Mshape"
trackable_dict_wrapper
+
	Eshape"
trackable_dict_wrapper
+
	Hshape"
trackable_dict_wrapper
+
	Fshape"
trackable_dict_wrapper
+
	Gshape"
trackable_dict_wrapper
+
	Ishape"
trackable_dict_wrapper
+
	Jshape"
trackable_dict_wrapper
Ћ
N	capture_0
O	capture_3
P	capture_5
Q	capture_7
R	capture_9
S
capture_10
T
capture_11
U
capture_12
V
capture_13
W
capture_14
X
capture_16
Y
capture_18
Z
capture_20
[
capture_21
\
capture_22
]
capture_23
^
capture_25
_
capture_27
`
capture_28
a
capture_29
b
capture_30B
__inference_compute_9731atomic_mu_ibatch_tot_natbatch_tot_nat_realbond_vectorind_iind_jmu_imu_j"
В
FullArgSpec
args
j
input_data
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zN	capture_0zO	capture_3zP	capture_5zQ	capture_7zR	capture_9zS
capture_10zT
capture_11zU
capture_12zV
capture_13zW
capture_14zX
capture_16zY
capture_18zZ
capture_20z[
capture_21z\
capture_22z]
capture_23z^
capture_25z_
capture_27z`
capture_28za
capture_29zb
capture_30

N	capture_0
O	capture_3
P	capture_5
Q	capture_7
R	capture_9
S
capture_10
T
capture_11
U
capture_12
V
capture_13
W
capture_14
X
capture_16
Y
capture_18
Z
capture_20
[
capture_21
\
capture_22
]
capture_23
^
capture_25
_
capture_27
`
capture_28
a
capture_29
b
capture_30Bѓ
"__inference_signature_wrapper_9812atomic_mu_ibatch_tot_natbatch_tot_nat_realbond_vectorind_iind_jmu_imu_j"ѕ
юВъ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 x

kwonlyargsjg
jatomic_mu_i
jbatch_tot_nat
jbatch_tot_nat_real
jbond_vector
jind_i
jind_j
jmu_i
jmu_j
kwonlydefaults
 
annotationsЊ *
 zN	capture_0zO	capture_3zP	capture_5zQ	capture_7zR	capture_9zS
capture_10zT
capture_11zU
capture_12zV
capture_13zW
capture_14zX
capture_16zY
capture_18zZ
capture_20z[
capture_21z\
capture_22z]
capture_23z^
capture_25z_
capture_27z`
capture_28za
capture_29zb
capture_30
 "
trackable_dict_wrapper
1
bond_length"
trackable_dict_wrapper
M
	bonds
celement_map
dcutoff_dict"
trackable_dict_wrapper
 "
trackable_dict_wrapper
:2
RBF_cutoff
+
		basis"
trackable_dict_wrapper
 "
trackable_list_wrapper
a
elayers_config

flayer0

glayer1

hlayer2

ilayer3"
_generic_user_object
*
vhat"
trackable_dict_wrapper
"
_generic_user_object
1
jelement_map"
trackable_dict_wrapper
:2element_map_symbols
:2element_map_index
%:#2Z/ChemicalEmbedding
H
	indicator


radial
angular"
trackable_dict_wrapper
%
kw"
_generic_user_object
O
linstructions

mls_max
nallowed_l_p"
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
o0"
trackable_list_wrapper
'
pA"
trackable_dict_wrapper
 "
trackable_dict_wrapper
(:&2rho/reducing_A
 "
trackable_dict_wrapper
K
qhidden_layers

rorigin

target"
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
I
slayers_config

tlayer0

ulayer1"
_generic_user_object
,

target"
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
trackable_list_wrapper
 "
trackable_list_wrapper
"J

Const_20jtf.TrackableConstant
"J

Const_19jtf.TrackableConstant
"J

Const_18jtf.TrackableConstant
"J

Const_17jtf.TrackableConstant
"J

Const_16jtf.TrackableConstant
"J

Const_15jtf.TrackableConstant
"J

Const_14jtf.TrackableConstant
"J

Const_13jtf.TrackableConstant
"J

Const_12jtf.TrackableConstant
"J

Const_11jtf.TrackableConstant
"J

Const_10jtf.TrackableConstant
!J	
Const_9jtf.TrackableConstant
!J	
Const_8jtf.TrackableConstant
!J	
Const_7jtf.TrackableConstant
!J	
Const_6jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
%
vw"
_generic_user_object
%
ww"
_generic_user_object
%
xw"
_generic_user_object
%
yw"
_generic_user_object
 "
trackable_dict_wrapper
::82(ChemIndTransf/DenseLayer_ChemIndTransf__
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
o0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
%
zw"
_generic_user_object
%
{w"
_generic_user_object
;:9@2)DenseLayer/DenseLayer_DenseLayer_no_decay
;:9@@2)DenseLayer/DenseLayer_DenseLayer_no_decay
;:9@@2)DenseLayer/DenseLayer_DenseLayer_no_decay
;:9@2)DenseLayer/DenseLayer_DenseLayer_no_decay
4:2@2"DenseLayer/DenseLayer_DenseLayer__
4:2@2"DenseLayer/DenseLayer_DenseLayer__
/b-
DenseLayer/Cast:0__inference_compute_9731
8b6
DenseLayer/einsum/Einsum:0__inference_compute_9731
1b/
DenseLayer/Cast_1:0__inference_compute_9731
:b8
DenseLayer/einsum_1/Einsum:0__inference_compute_9731
1b/
DenseLayer/Cast_2:0__inference_compute_9731
:b8
DenseLayer/einsum_2/Einsum:0__inference_compute_9731
1b/
DenseLayer/Cast_3:0__inference_compute_9731
:b8
DenseLayer/einsum_4/Einsum:0__inference_compute_9731э
__inference_compute_9731аN*vOwPxQyRSTUVWkX3Y<Z[\]z^{_`abяЂы
уЂп
мЊи
0
atomic_mu_i!
atomic_mu_iџџџџџџџџџ
1
batch_tot_nat_real
batch_tot_nat_real 
'
batch_tot_nat
batch_tot_nat 
4
bond_vector%"
bond_vectorџџџџџџџџџ
$
ind_i
ind_iџџџџџџџџџ
$
ind_j
ind_jџџџџџџџџџ
"
mu_i
mu_iџџџџџџџџџ
"
mu_j
mu_jџџџџџџџџџ
Њ "КЊЖ
8
atomic_energy'$
atomic_energyџџџџџџџџџ
-
total_energy
total_energy
,
total_f!
total_fџџџџџџџџџ

virial
virialъ
!__inference_internal_grad_fn_9897Ф|}~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 ъ
!__inference_internal_grad_fn_9924Ф~~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 ь
!__inference_internal_grad_fn_9951Ц~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 ь
!__inference_internal_grad_fn_9978Ц~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 №
"__inference_signature_wrapper_9812ЩN*vOwPxQyRSTUVWkX3Y<Z[\]z^{_`abшЂф
Ђ 
мЊи
0
atomic_mu_i!
atomic_mu_iџџџџџџџџџ
1
batch_tot_nat_real
batch_tot_nat_real 
'
batch_tot_nat
batch_tot_nat 
4
bond_vector%"
bond_vectorџџџџџџџџџ
$
ind_i
ind_iџџџџџџџџџ
$
ind_j
ind_jџџџџџџџџџ
"
mu_i
mu_iџџџџџџџџџ
"
mu_j
mu_jџџџџџџџџџ"КЊЖ
8
atomic_energy'$
atomic_energyџџџџџџџџџ
-
total_energy
total_energy
,
total_f!
total_fџџџџџџџџџ

virial
virial