ц
Є+ј*
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
,
Cos
x"T
y"T"
Ttype:

2
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
=
Greater
x"T
y"T
z
"
Ttype:
2	
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
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
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
,
Sin
x"T
y"T"
Ttype:

2
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
Ttype"serve*2.15.02v2.15.0-2-g0b15fdfcb3f8У
N
ConstConst*
_output_shapes
: *
dtype0*
valueB 2      Р?
P
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2      Р?
P
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2Ь;f ц?
P
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2-DTћ!	@
P
Const_4Const*
_output_shapes
: *
dtype0*
valueB 2:0тyE>
P
Const_5Const*
_output_shapes
: *
dtype0*
valueB 23ЇЈе#іI9
P
Const_6Const*
_output_shapes
: *
dtype0*
valueB 2Zй­фKм?
P
Const_7Const*
_output_shapes
: *
dtype0*
valueB 2Ь;f Ц?
P
Const_8Const*
_output_shapes
: *
dtype0*
valueB 2      №?
P
Const_9Const*
_output_shapes
: *
dtype0*
valueB 2        
Z
Const_10Const*
_output_shapes

:*
dtype0*
valueB: 
Q
Const_11Const*
_output_shapes
: *
dtype0*
valueB 2      №?
R
Const_12Const*
_output_shapes
:*
dtype0*
valueB: 
Z
Const_13Const*
_output_shapes

:*
dtype0*
valueB: 
Q
Const_14Const*
_output_shapes
: *
dtype0*
valueB 2      №?
R
Const_15Const*
_output_shapes
:*
dtype0*
valueB: 
­
Const_16Const*
_output_shapes

:*
dtype0*m
valuedBb"T                                                                  
Q
Const_17Const*
_output_shapes
: *
dtype0*
valueB 2      №?
Ѕ
Const_18Const*
_output_shapes
:*
dtype0*i
value`B^"T                                                                  
Q
Const_19Const*
_output_shapes
: *
dtype0*
valueB 2      №?
J
Const_20Const*
_output_shapes
: *
dtype0*
value	B :
щ
Const_21Const*
_output_shapes
:%*
dtype0*Ќ
valueЂB%"                                           	   	   	   
   
   
   
                                                   

Const_22Const*"
_output_shapes
:%*
dtype0*Ш
valueОBЛ%"Ј      №?3EЇyт?3EЇyт?3EЇyт?ўџџџџџя?      №?ўџџџџџя?ўџџџџџя?      №?ўџџџџџя?qјtёс?qјtёс?IHb=дПqјtёсПqјtёс?IHb=ф?qјtёс?qјtёс?IHb=дПqјtёс?qјtёс?Ы;f ц?Ы;f ц?Ь;f ц?Ь;f ц?=,pН кП>,pН ъ?=,pН кПЬ;f ц?Ь;f ц?Ы;f цПЫ;f ц?ўџџџџџя?ўџџџџџя?      №?ўџџџџџя?ўџџџџџя?
щ
Const_23Const*
_output_shapes
:%*
dtype0*Ќ
valueЂB%"                                                                                                                   
щ
Const_24Const*
_output_shapes
:%*
dtype0*Ќ
valueЂB%"                                                                                                                         
Z
Const_25Const*
_output_shapes

:*
dtype0*
valueB: 
Q
Const_26Const*
_output_shapes
: *
dtype0*
valueB 2      №?
R
Const_27Const*
_output_shapes
:*
dtype0*
valueB: 
Z
Const_28Const*
_output_shapes

:*
dtype0*
valueB: 
Q
Const_29Const*
_output_shapes
: *
dtype0*
valueB 2      №?
R
Const_30Const*
_output_shapes
:*
dtype0*
valueB: 
i
Const_31Const*
_output_shapes

:*
dtype0*)
value B"             
Q
Const_32Const*
_output_shapes
: *
dtype0*
valueB 2      №?
a
Const_33Const*
_output_shapes
:*
dtype0*%
valueB"               
Q
Const_34Const*
_output_shapes
: *
dtype0*
valueB 2      №?
Q
Const_35Const*
_output_shapes
: *
dtype0*
valueB 2Ь;f ц?
Q
Const_36Const*
_output_shapes
: *
dtype0*
valueB 2jяДј[@

Const_37Const*
_output_shapes
:*
dtype0*E
value<B:"0                        3EЇyтП               

Const_38Const*
_output_shapes
:*
dtype0*E
value<B:"0        ЊLXшzЖћ?.!	ѓПкNOБоћў?Јєwу@ЈєwуёП
Q
Const_39Const*
_output_shapes
: *
dtype0*
valueB 2-DTћ!	@
u
Const_40Const*
_output_shapes
:	*
dtype0*9
value0B.	"$                            
Q
Const_41Const*
_output_shapes
: *
dtype0*
valueB 2      Р?
е
"DenseLayer/DenseLayer_DenseLayer__VarHandleOp*
_output_shapes
: *3

debug_name%#DenseLayer/DenseLayer_DenseLayer__/*
dtype0*
shape
: *3
shared_name$"DenseLayer/DenseLayer_DenseLayer__

6DenseLayer/DenseLayer_DenseLayer__/Read/ReadVariableOpReadVariableOp"DenseLayer/DenseLayer_DenseLayer__*
_output_shapes

: *
dtype0
л
$DenseLayer/DenseLayer_DenseLayer___1VarHandleOp*
_output_shapes
: *5

debug_name'%DenseLayer/DenseLayer_DenseLayer___1/*
dtype0*
shape
: *5
shared_name&$DenseLayer/DenseLayer_DenseLayer___1

8DenseLayer/DenseLayer_DenseLayer___1/Read/ReadVariableOpReadVariableOp$DenseLayer/DenseLayer_DenseLayer___1*
_output_shapes

: *
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
y
cutoffVarHandleOp*
_output_shapes
: *

debug_name	cutoff/*
dtype0*
shape: *
shared_namecutoff
Y
cutoff/Read/ReadVariableOpReadVariableOpcutoff*
_output_shapes
: *
dtype0
Є
rho2/reducing_BVarHandleOp*
_output_shapes
: * 

debug_namerho2/reducing_B/*
dtype0*
shape:* 
shared_namerho2/reducing_B
{
#rho2/reducing_B/Read/ReadVariableOpReadVariableOprho2/reducing_B*&
_output_shapes
:*
dtype0

E2/reducing_BVarHandleOp*
_output_shapes
: *

debug_nameE2/reducing_B/*
dtype0*
shape:*
shared_nameE2/reducing_B
w
!E2/reducing_B/Read/ReadVariableOpReadVariableOpE2/reducing_B*&
_output_shapes
:*
dtype0

B/reducing_YIVarHandleOp*
_output_shapes
: *

debug_nameB/reducing_YI/*
dtype0*
shape:*
shared_nameB/reducing_YI
s
!B/reducing_YI/Read/ReadVariableOpReadVariableOpB/reducing_YI*"
_output_shapes
:*
dtype0
Ё
rho/reducing_AVarHandleOp*
_output_shapes
: *

debug_namerho/reducing_A/*
dtype0*
shape:*
shared_namerho/reducing_A
y
"rho/reducing_A/Read/ReadVariableOpReadVariableOprho/reducing_A*&
_output_shapes
:*
dtype0

E/reducing_AVarHandleOp*
_output_shapes
: *

debug_nameE/reducing_A/*
dtype0*
shape:*
shared_nameE/reducing_A
u
 E/reducing_A/Read/ReadVariableOpReadVariableOpE/reducing_A*&
_output_shapes
:*
dtype0

I/reducing_AVarHandleOp*
_output_shapes
: *

debug_nameI/reducing_A/*
dtype0*
shape:*
shared_nameI/reducing_A
u
 I/reducing_A/Read/ReadVariableOpReadVariableOpI/reducing_A*&
_output_shapes
:*
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
Џ
StatefulPartitionedCallStatefulPartitionedCallserving_default_atomic_mu_iserving_default_batch_tot_nat"serving_default_batch_tot_nat_realserving_default_bond_vectorserving_default_ind_iserving_default_ind_jserving_default_mu_iserving_default_mu_jConst_5Const_4Const_3cutoff+DenseLayer/DenseLayer_DenseLayer_no_decay_3Const_2+DenseLayer/DenseLayer_DenseLayer_no_decay_2Const_1+DenseLayer/DenseLayer_DenseLayer_no_decay_1Const)DenseLayer/DenseLayer_DenseLayer_no_decayConst_41Const_40Const_39Const_38Const_37Const_36(ChemIndTransf/DenseLayer_ChemIndTransf__Const_35Z/ChemicalEmbeddingConst_34I/reducing_AConst_33Const_32Const_31E/reducing_AConst_30Const_29Const_28rho/reducing_AConst_27Const_26Const_25Const_24Const_23Const_22Const_21Const_20Const_19B/reducing_YIConst_18Const_17Const_16E2/reducing_BConst_15Const_14Const_13rho2/reducing_BConst_12Const_11Const_10Const_9$DenseLayer/DenseLayer_DenseLayer___1Const_8"DenseLayer/DenseLayer_DenseLayer__Const_7Const_6*L
TinE
C2A*
Tout	
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ::џџџџџџџџџ::џџџџџџџџџ*1
_read_only_resource_inputs
!%/37<>*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_143490

NoOpNoOp
Ѓ:
Const_42Const"/device:CPU:0*
_output_shapes
: *
dtype0*л9
valueб9BЮ9 BЧ9
W
instructions
compute_specs
train_specs
compute

signatures*

0
1
2
	3

4
5
6
7
8
9
10
11
12
13
14
15
16
17*
y
	ind_i
	ind_j
bond_vector
mu_i
mu_j
atomic_mu_i
batch_tot_nat
batch_tot_nat_real* 
Љ
	 ind_i
	!ind_j
"map_atoms_to_structure
#n_struct_total
$bond_vector
%batch_tot_nat
&mu_i
'mu_j
(atomic_mu_i
)batch_tot_nat_real* 

*trace_0* 

+serving_default* 

,
_init_args* 

-
_init_args* 
4
.
_init_args

/kwargs
0basis_function*
0
1
_init_args
2hidden_layers
3mlp*

4
_init_args
5sg* 
K
6
_init_args
7element_map_symbols
8element_map_index
9w*
O
:
_init_args

	radial

angular
	indicator
;lin_transform*
|
<
_init_args
=instructions

>ls_max
?allowed_l_p
@	collector
Adownscale_embeddings
B
reducing_A*
|
C
_init_args
Dinstructions

Els_max
Fallowed_l_p
G	collector
Hdownscale_embeddings
I
reducing_A*
|
J
_init_args
Kinstructions

Lls_max
Mallowed_l_p
N	collector
Odownscale_embeddings
P
reducing_A*
Q
Q
_init_args

	radial

angular
	indicator
Rcoupling_origin*
}
S
_init_args
Tinstructions

Uls_max
Vallowed_l_p
W	collector
Xdownscale_embeddings
Yreducing_YI*
|
Z
_init_args
[instructions

\ls_max
]allowed_l_p
^	collector
_downscale_embeddings
`
reducing_B*
|
a
_init_args
binstructions

cls_max
dallowed_l_p
e	collector
fdownscale_embeddings
g
reducing_B*

h
_init_args* 
,
i
_init_args

target

jorigin*
H
k
_init_args

target

lorigin
mhidden_layers
nmlp*

o
_init_args

target* 

	pshape* 

	qshape* 

	rshape* 

	sshape* 

	tshape* 

	ushape* 

	vshape* 

	wshape* 

	pshape* 

	qshape* 

	xshape* 

	yshape* 

	rshape* 

	vshape* 

	sshape* 

	tshape* 

	ushape* 

	wshape* 
Р
z	capture_0
{	capture_1
|	capture_2
}	capture_5
~	capture_7
	capture_9

capture_11

capture_12

capture_13

capture_14

capture_15

capture_16

capture_18

capture_20

capture_22

capture_23

capture_24

capture_26

capture_27

capture_28

capture_30

capture_31

capture_32

capture_33

capture_34

capture_35

capture_36

capture_37

capture_38

capture_40

capture_41

capture_42

capture_44

capture_45

capture_46

capture_48

capture_49

capture_50
 
capture_51
Ё
capture_53
Ђ
capture_55
Ѓ
capture_56* 
Р
z	capture_0
{	capture_1
|	capture_2
}	capture_5
~	capture_7
	capture_9

capture_11

capture_12

capture_13

capture_14

capture_15

capture_16

capture_18

capture_20

capture_22

capture_23

capture_24

capture_26

capture_27

capture_28

capture_30

capture_31

capture_32

capture_33

capture_34

capture_35

capture_36

capture_37

capture_38

capture_40

capture_41

capture_42

capture_44

capture_45

capture_46

capture_48

capture_49

capture_50
 
capture_51
Ё
capture_53
Ђ
capture_55
Ѓ
capture_56* 
* 

bond_length* 

	bonds* 
* 

Єrc*

	basis*
* 
L
Ѕlayers_config
Іlayer0
Їlayer1
Јlayer2
Љlayer3*

vhat* 
* 

Њelement_map* 
jd
VARIABLE_VALUEelement_map_symbols=instructions/5/element_map_symbols/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEelement_map_index;instructions/5/element_map_index/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEZ/ChemicalEmbedding+instructions/5/w/.ATTRIBUTES/VARIABLE_VALUE*
,
	indicator

angular

	radial*

Ћw*
6
Ќallowed_l_p
­instructions
Ўls_max*

0*
* 

Џ0
А1* 


БA* 
* 
ZT
VARIABLE_VALUEI/reducing_A4instructions/7/reducing_A/.ATTRIBUTES/VARIABLE_VALUE*
6
Вallowed_l_p
Гinstructions
Дls_max*

0*
* 


Е0* 


ЖA* 
* 
ZT
VARIABLE_VALUEE/reducing_A4instructions/8/reducing_A/.ATTRIBUTES/VARIABLE_VALUE*
6
Зallowed_l_p
Иinstructions
Йls_max*

0*
* 


К0* 


ЛA* 
* 
\V
VARIABLE_VALUErho/reducing_A4instructions/9/reducing_A/.ATTRIBUTES/VARIABLE_VALUE*
>

	radial
Мkeep_parity

angular
	indicator*
* 
6
Нallowed_l_p
Оinstructions
Пls_max*

0*
* 
:
Р0
С1
Т2
У3
Ф4
Х5
Ц6* 

ЧYI* 
* 
]W
VARIABLE_VALUEB/reducing_YI6instructions/11/reducing_YI/.ATTRIBUTES/VARIABLE_VALUE*
6
Шallowed_l_p
Щinstructions
Ъls_max*

0*
* 


Ы0* 


ЬB* 
* 
\V
VARIABLE_VALUEE2/reducing_B5instructions/12/reducing_B/.ATTRIBUTES/VARIABLE_VALUE*
6
Эallowed_l_p
Юinstructions
Яls_max*

0*
* 


а0* 


бB* 
* 
^X
VARIABLE_VALUErho2/reducing_B5instructions/13/reducing_B/.ATTRIBUTES/VARIABLE_VALUE*
* 

вorigin

target*

0
1*

гorigin

target*

0
1*
* 
2
дlayers_config
еlayer0
жlayer1*


target* 
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
[U
VARIABLE_VALUEcutoff;instructions/2/basis_function/rc/.ATTRIBUTES/VARIABLE_VALUE*
* 

зw*

иw*

йw*

кw*
* 
{u
VARIABLE_VALUE(ChemIndTransf/DenseLayer_ChemIndTransf__9instructions/6/lin_transform/w/.ATTRIBUTES/VARIABLE_VALUE*

л0
м1* 

0*
* 
* 
* 
* 


н0* 

0*
* 
* 
* 


о0* 

0*
* 
* 
* 
:
п0
р1
с2
т3
у4
ф5
х6* 
:
ц0
ч1
ш2
щ3
ъ4
ы5
ь6* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 


э0* 

0*
* 
* 
* 


ю0* 

0*
* 
* 
* 

0
1*

0
1*
* 

яw*

№w*
{u
VARIABLE_VALUE+DenseLayer/DenseLayer_DenseLayer_no_decay_36instructions/3/mlp/layer0/w/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE+DenseLayer/DenseLayer_DenseLayer_no_decay_26instructions/3/mlp/layer1/w/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE+DenseLayer/DenseLayer_DenseLayer_no_decay_16instructions/3/mlp/layer2/w/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE)DenseLayer/DenseLayer_DenseLayer_no_decay6instructions/3/mlp/layer3/w/.ATTRIBUTES/VARIABLE_VALUE*
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
uo
VARIABLE_VALUE$DenseLayer/DenseLayer_DenseLayer___17instructions/16/mlp/layer0/w/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE"DenseLayer/DenseLayer_DenseLayer__7instructions/16/mlp/layer1/w/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ѕ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameelement_map_symbolselement_map_indexZ/ChemicalEmbeddingI/reducing_AE/reducing_Arho/reducing_AB/reducing_YIE2/reducing_Brho2/reducing_Bcutoff(ChemIndTransf/DenseLayer_ChemIndTransf__+DenseLayer/DenseLayer_DenseLayer_no_decay_3+DenseLayer/DenseLayer_DenseLayer_no_decay_2+DenseLayer/DenseLayer_DenseLayer_no_decay_1)DenseLayer/DenseLayer_DenseLayer_no_decay$DenseLayer/DenseLayer_DenseLayer___1"DenseLayer/DenseLayer_DenseLayer__Const_42*
Tin
2*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_143775
э
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameelement_map_symbolselement_map_indexZ/ChemicalEmbeddingI/reducing_AE/reducing_Arho/reducing_AB/reducing_YIE2/reducing_Brho2/reducing_Bcutoff(ChemIndTransf/DenseLayer_ChemIndTransf__+DenseLayer/DenseLayer_DenseLayer_no_decay_3+DenseLayer/DenseLayer_DenseLayer_no_decay_2+DenseLayer/DenseLayer_DenseLayer_no_decay_1)DenseLayer/DenseLayer_DenseLayer_no_decay$DenseLayer/DenseLayer_DenseLayer___1"DenseLayer/DenseLayer_DenseLayer__*
Tin
2*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_143835ќ
в
О
#__inference_internal_grad_fn_143661
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
Р
К
#__inference_internal_grad_fn_143607
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
2

$__inference_signature_wrapper_143490
atomic_mu_i
batch_tot_nat
batch_tot_nat_real
bond_vector	
ind_i	
ind_j
mu_i
mu_j
unknown
	unknown_0
	unknown_1
	unknown_2: 
	unknown_3:@
	unknown_4
	unknown_5:@@
	unknown_6
	unknown_7:@@
	unknown_8
	unknown_9:@

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16:

unknown_17

unknown_18:

unknown_19$

unknown_20:

unknown_21

unknown_22

unknown_23$

unknown_24:

unknown_25

unknown_26

unknown_27$

unknown_28:

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37 

unknown_38:

unknown_39

unknown_40

unknown_41$

unknown_42:

unknown_43

unknown_44

unknown_45$

unknown_46:

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51: 

unknown_52

unknown_53: 

unknown_54

unknown_55
identity

identity_1

identity_2

identity_3

identity_4ЂStatefulPartitionedCallЅ	
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
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55*L
TinE
C2A*
Tout	
2*
_XlaMustCompile( *
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ::џџџџџџџџџ::џџџџџџџџџ*1
_read_only_resource_inputs
!%/37<>*-
config_proto

CPU

GPU 2J 8 *#
fR
__inference_compute_143355o
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
:q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*е
_input_shapesУ
Р:џџџџџџџџџ: : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : :	: ::: : : : : : :: :: :: :: :: ::%:%:%:%: : : :: :: :: :: :: :: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:@

_output_shapes
: :?

_output_shapes
: :&>"
 
_user_specified_name143474:=

_output_shapes
: :&<"
 
_user_specified_name143470:;

_output_shapes
: :$: 

_output_shapes

::9

_output_shapes
: : 8

_output_shapes
::&7"
 
_user_specified_name143460:$6 

_output_shapes

::5

_output_shapes
: : 4

_output_shapes
::&3"
 
_user_specified_name143452:$2 

_output_shapes

::1

_output_shapes
: : 0

_output_shapes
::&/"
 
_user_specified_name143444:.

_output_shapes
: :-

_output_shapes
: : ,

_output_shapes
:%:(+$
"
_output_shapes
:%: *

_output_shapes
:%: )

_output_shapes
:%:$( 

_output_shapes

::'

_output_shapes
: : &

_output_shapes
::&%"
 
_user_specified_name143424:$$ 

_output_shapes

::#

_output_shapes
: : "

_output_shapes
::&!"
 
_user_specified_name143416:$  

_output_shapes

::

_output_shapes
: : 

_output_shapes
::&"
 
_user_specified_name143408:

_output_shapes
: :&"
 
_user_specified_name143404:

_output_shapes
: :&"
 
_user_specified_name143400:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:	:

_output_shapes
: :&"
 
_user_specified_name143386:

_output_shapes
: :&"
 
_user_specified_name143382:

_output_shapes
: :&"
 
_user_specified_name143378:

_output_shapes
: :&"
 
_user_specified_name143374:&"
 
_user_specified_name143372:


_output_shapes
: :	

_output_shapes
: :
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
яЋ

__inference_compute_143355
atomic_mu_i
batch_tot_nat
batch_tot_nat_real
bond_vector	
ind_i	
ind_j
mu_i
mu_j
scaledbondvector_add_y-
)simplifiedbesselradialbasisfunction_add_y/
+simplifiedbesselradialbasisfunction_mul_1_yK
Asimplifiedbesselradialbasisfunction_pow_1_readvariableop_resource: 4
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
y_140819
y_140826

y_mul_35_y7
%chemindtransf_readvariableop_resource:
chemindtransf_mul_yE
3chemindtransf_einsum_einsum_readvariableop_resource:
a_mul_y>
$i_gatherv2_1_readvariableop_resource:
i_gatherv2_1_indices
i_mul_y
i_tensorscatteradd_indices>
$e_gatherv2_1_readvariableop_resource:
e_gatherv2_1_indices
e_mul_y
e_tensorscatteradd_indices@
&rho_gatherv2_1_readvariableop_resource:
rho_gatherv2_1_indices
	rho_mul_y 
rho_tensorscatteradd_indices
yi_gatherv2_1_indices
yi_gatherv2_2_indices

yi_mul_1_y
yi_sum_cg_yi_segment_ids
yi_sum_cg_yi_num_segments

yi_mul_2_y:
$b_gatherv2_1_readvariableop_resource:
b_gatherv2_1_indices
b_mul_y
b_tensorscatteradd_indices?
%e2_gatherv2_1_readvariableop_resource:
e2_gatherv2_1_indices
e2_mul_y
e2_tensorscatteradd_indicesA
'rho2_gatherv2_1_readvariableop_resource:
rho2_gatherv2_1_indices

rho2_mul_y!
rho2_tensorscatteradd_indices
add_2_x6
$denselayer_readvariableop_4_resource: 
denselayer_mul_13_y6
$denselayer_readvariableop_5_resource: 
denselayer_mul_17_y	
mul_y
identity

identity_1

identity_2

identity_3

identity_4ЂB/GatherV2_1/ReadVariableOpЂChemIndTransf/ReadVariableOpЂ*ChemIndTransf/einsum/Einsum/ReadVariableOpЂDenseLayer/ReadVariableOpЂDenseLayer/ReadVariableOp_1ЂDenseLayer/ReadVariableOp_2ЂDenseLayer/ReadVariableOp_3ЂDenseLayer/ReadVariableOp_4ЂDenseLayer/ReadVariableOp_5ЂE/GatherV2_1/ReadVariableOpЂE2/GatherV2_1/ReadVariableOpЂI/GatherV2_1/ReadVariableOpЂ8SimplifiedBesselRadialBasisFunction/Pow_1/ReadVariableOpЂ8SimplifiedBesselRadialBasisFunction/Pow_8/ReadVariableOpЂ=SimplifiedBesselRadialBasisFunction/truediv_11/ReadVariableOpЂ=SimplifiedBesselRadialBasisFunction/truediv_13/ReadVariableOpЂ<SimplifiedBesselRadialBasisFunction/truediv_2/ReadVariableOpЂ<SimplifiedBesselRadialBasisFunction/truediv_4/ReadVariableOpЂrho/GatherV2_1/ReadVariableOpЂrho2/GatherV2_1/ReadVariableOpf
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
:џџџџџџџџџt
+SimplifiedBesselRadialBasisFunction/Equal/yConst*
_output_shapes
: *
dtype0*
valueB 2        д
)SimplifiedBesselRadialBasisFunction/EqualEqualBondLength/norm/Sqrt:y:04SimplifiedBesselRadialBasisFunction/Equal/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
incompatible_shape_error( Ї
'SimplifiedBesselRadialBasisFunction/addAddV2BondLength/norm/Sqrt:y:0)simplifiedbesselradialbasisfunction_add_y*
T0*'
_output_shapes
:џџџџџџџџџр
,SimplifiedBesselRadialBasisFunction/SelectV2SelectV2-SimplifiedBesselRadialBasisFunction/Equal:z:0+SimplifiedBesselRadialBasisFunction/add:z:0BondLength/norm/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
)SimplifiedBesselRadialBasisFunction/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        t
+SimplifiedBesselRadialBasisFunction/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2      №ПЙ
'SimplifiedBesselRadialBasisFunction/PowPow4SimplifiedBesselRadialBasisFunction/Const_1:output:02SimplifiedBesselRadialBasisFunction/Const:output:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2       @
(SimplifiedBesselRadialBasisFunction/SqrtSqrt4SimplifiedBesselRadialBasisFunction/Const_2:output:0*
T0*
_output_shapes
: Њ
'SimplifiedBesselRadialBasisFunction/mulMul+SimplifiedBesselRadialBasisFunction/Pow:z:0,SimplifiedBesselRadialBasisFunction/Sqrt:y:0*
T0*
_output_shapes
: Ћ
)SimplifiedBesselRadialBasisFunction/mul_1Mul+SimplifiedBesselRadialBasisFunction/mul:z:0+simplifiedbesselradialbasisfunction_mul_1_y*
T0*
_output_shapes
: В
8SimplifiedBesselRadialBasisFunction/Pow_1/ReadVariableOpReadVariableOpAsimplifiedbesselradialbasisfunction_pow_1_readvariableop_resource*
_output_shapes
: *
dtype0t
+SimplifiedBesselRadialBasisFunction/Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB 2      ј?Щ
)SimplifiedBesselRadialBasisFunction/Pow_1Pow@SimplifiedBesselRadialBasisFunction/Pow_1/ReadVariableOp:value:04SimplifiedBesselRadialBasisFunction/Pow_1/y:output:0*
T0*
_output_shapes
: Е
+SimplifiedBesselRadialBasisFunction/truedivRealDiv-SimplifiedBesselRadialBasisFunction/mul_1:z:0-SimplifiedBesselRadialBasisFunction/Pow_1:z:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/add_1/yConst*
_output_shapes
: *
dtype0*
valueB 2      №?Н
)SimplifiedBesselRadialBasisFunction/add_1AddV22SimplifiedBesselRadialBasisFunction/Const:output:04SimplifiedBesselRadialBasisFunction/add_1/y:output:0*
T0*
_output_shapes
: Б
)SimplifiedBesselRadialBasisFunction/mul_2Mul/SimplifiedBesselRadialBasisFunction/truediv:z:0-SimplifiedBesselRadialBasisFunction/add_1:z:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/add_2/yConst*
_output_shapes
: *
dtype0*
valueB 2       @Н
)SimplifiedBesselRadialBasisFunction/add_2AddV22SimplifiedBesselRadialBasisFunction/Const:output:04SimplifiedBesselRadialBasisFunction/add_2/y:output:0*
T0*
_output_shapes
: Џ
)SimplifiedBesselRadialBasisFunction/mul_3Mul-SimplifiedBesselRadialBasisFunction/mul_2:z:0-SimplifiedBesselRadialBasisFunction/add_2:z:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/add_3/yConst*
_output_shapes
: *
dtype0*
valueB 2      №?Н
)SimplifiedBesselRadialBasisFunction/add_3AddV22SimplifiedBesselRadialBasisFunction/Const:output:04SimplifiedBesselRadialBasisFunction/add_3/y:output:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB 2       @Ж
)SimplifiedBesselRadialBasisFunction/pow_2Pow-SimplifiedBesselRadialBasisFunction/add_3:z:04SimplifiedBesselRadialBasisFunction/pow_2/y:output:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/add_4/yConst*
_output_shapes
: *
dtype0*
valueB 2       @Н
)SimplifiedBesselRadialBasisFunction/add_4AddV22SimplifiedBesselRadialBasisFunction/Const:output:04SimplifiedBesselRadialBasisFunction/add_4/y:output:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB 2       @Ж
)SimplifiedBesselRadialBasisFunction/pow_3Pow-SimplifiedBesselRadialBasisFunction/add_4:z:04SimplifiedBesselRadialBasisFunction/pow_3/y:output:0*
T0*
_output_shapes
: Б
)SimplifiedBesselRadialBasisFunction/add_5AddV2-SimplifiedBesselRadialBasisFunction/pow_2:z:0-SimplifiedBesselRadialBasisFunction/pow_3:z:0*
T0*
_output_shapes
: 
*SimplifiedBesselRadialBasisFunction/Sqrt_1Sqrt-SimplifiedBesselRadialBasisFunction/add_5:z:0*
T0*
_output_shapes
: И
-SimplifiedBesselRadialBasisFunction/truediv_1RealDiv-SimplifiedBesselRadialBasisFunction/mul_3:z:0.SimplifiedBesselRadialBasisFunction/Sqrt_1:y:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/add_6/yConst*
_output_shapes
: *
dtype0*
valueB 2      №?Н
)SimplifiedBesselRadialBasisFunction/add_6AddV22SimplifiedBesselRadialBasisFunction/Const:output:04SimplifiedBesselRadialBasisFunction/add_6/y:output:0*
T0*
_output_shapes
: Ш
)SimplifiedBesselRadialBasisFunction/mul_4Mul5SimplifiedBesselRadialBasisFunction/SelectV2:output:0-SimplifiedBesselRadialBasisFunction/add_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџО
)SimplifiedBesselRadialBasisFunction/mul_5Mul-SimplifiedBesselRadialBasisFunction/mul_4:z:0+simplifiedbesselradialbasisfunction_mul_1_y*
T0*'
_output_shapes
:џџџџџџџџџЖ
<SimplifiedBesselRadialBasisFunction/truediv_2/ReadVariableOpReadVariableOpAsimplifiedbesselradialbasisfunction_pow_1_readvariableop_resource*
_output_shapes
: *
dtype0п
-SimplifiedBesselRadialBasisFunction/truediv_2RealDiv-SimplifiedBesselRadialBasisFunction/mul_5:z:0DSimplifiedBesselRadialBasisFunction/truediv_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџw
.SimplifiedBesselRadialBasisFunction/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB 2        і
,SimplifiedBesselRadialBasisFunction/NotEqualNotEqual1SimplifiedBesselRadialBasisFunction/truediv_2:z:07SimplifiedBesselRadialBasisFunction/NotEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
incompatible_shape_error( 
'SimplifiedBesselRadialBasisFunction/SinSin1SimplifiedBesselRadialBasisFunction/truediv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџЪ
-SimplifiedBesselRadialBasisFunction/truediv_3RealDiv+SimplifiedBesselRadialBasisFunction/Sin:y:01SimplifiedBesselRadialBasisFunction/truediv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
-SimplifiedBesselRadialBasisFunction/ones_likeOnesLike1SimplifiedBesselRadialBasisFunction/truediv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
.SimplifiedBesselRadialBasisFunction/SelectV2_1SelectV20SimplifiedBesselRadialBasisFunction/NotEqual:z:01SimplifiedBesselRadialBasisFunction/truediv_3:z:01SimplifiedBesselRadialBasisFunction/ones_like:y:0*
T0*'
_output_shapes
:џџџџџџџџџt
+SimplifiedBesselRadialBasisFunction/add_7/yConst*
_output_shapes
: *
dtype0*
valueB 2       @Н
)SimplifiedBesselRadialBasisFunction/add_7AddV22SimplifiedBesselRadialBasisFunction/Const:output:04SimplifiedBesselRadialBasisFunction/add_7/y:output:0*
T0*
_output_shapes
: Ш
)SimplifiedBesselRadialBasisFunction/mul_6Mul5SimplifiedBesselRadialBasisFunction/SelectV2:output:0-SimplifiedBesselRadialBasisFunction/add_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџО
)SimplifiedBesselRadialBasisFunction/mul_7Mul-SimplifiedBesselRadialBasisFunction/mul_6:z:0+simplifiedbesselradialbasisfunction_mul_1_y*
T0*'
_output_shapes
:џџџџџџџџџЖ
<SimplifiedBesselRadialBasisFunction/truediv_4/ReadVariableOpReadVariableOpAsimplifiedbesselradialbasisfunction_pow_1_readvariableop_resource*
_output_shapes
: *
dtype0п
-SimplifiedBesselRadialBasisFunction/truediv_4RealDiv-SimplifiedBesselRadialBasisFunction/mul_7:z:0DSimplifiedBesselRadialBasisFunction/truediv_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџy
0SimplifiedBesselRadialBasisFunction/NotEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB 2        њ
.SimplifiedBesselRadialBasisFunction/NotEqual_1NotEqual1SimplifiedBesselRadialBasisFunction/truediv_4:z:09SimplifiedBesselRadialBasisFunction/NotEqual_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
incompatible_shape_error( 
)SimplifiedBesselRadialBasisFunction/Sin_1Sin1SimplifiedBesselRadialBasisFunction/truediv_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџЬ
-SimplifiedBesselRadialBasisFunction/truediv_5RealDiv-SimplifiedBesselRadialBasisFunction/Sin_1:y:01SimplifiedBesselRadialBasisFunction/truediv_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 
/SimplifiedBesselRadialBasisFunction/ones_like_1OnesLike1SimplifiedBesselRadialBasisFunction/truediv_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
.SimplifiedBesselRadialBasisFunction/SelectV2_2SelectV22SimplifiedBesselRadialBasisFunction/NotEqual_1:z:01SimplifiedBesselRadialBasisFunction/truediv_5:z:03SimplifiedBesselRadialBasisFunction/ones_like_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџж
)SimplifiedBesselRadialBasisFunction/add_8AddV27SimplifiedBesselRadialBasisFunction/SelectV2_1:output:07SimplifiedBesselRadialBasisFunction/SelectV2_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџФ
)SimplifiedBesselRadialBasisFunction/mul_8Mul1SimplifiedBesselRadialBasisFunction/truediv_1:z:0-SimplifiedBesselRadialBasisFunction/add_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџt
+SimplifiedBesselRadialBasisFunction/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2      №?t
+SimplifiedBesselRadialBasisFunction/Const_4Const*
_output_shapes
: *
dtype0*
valueB 2      №?t
+SimplifiedBesselRadialBasisFunction/pow_4/yConst*
_output_shapes
: *
dtype0*
valueB 2       @Н
)SimplifiedBesselRadialBasisFunction/pow_4Pow4SimplifiedBesselRadialBasisFunction/Const_4:output:04SimplifiedBesselRadialBasisFunction/pow_4/y:output:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/add_9/yConst*
_output_shapes
: *
dtype0*
valueB 2       @П
)SimplifiedBesselRadialBasisFunction/add_9AddV24SimplifiedBesselRadialBasisFunction/Const_4:output:04SimplifiedBesselRadialBasisFunction/add_9/y:output:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/pow_5/yConst*
_output_shapes
: *
dtype0*
valueB 2       @Ж
)SimplifiedBesselRadialBasisFunction/pow_5Pow-SimplifiedBesselRadialBasisFunction/add_9:z:04SimplifiedBesselRadialBasisFunction/pow_5/y:output:0*
T0*
_output_shapes
: Џ
)SimplifiedBesselRadialBasisFunction/mul_9Mul-SimplifiedBesselRadialBasisFunction/pow_4:z:0-SimplifiedBesselRadialBasisFunction/pow_5:z:0*
T0*
_output_shapes
: u
,SimplifiedBesselRadialBasisFunction/add_10/yConst*
_output_shapes
: *
dtype0*
valueB 2      №?С
*SimplifiedBesselRadialBasisFunction/add_10AddV24SimplifiedBesselRadialBasisFunction/Const_4:output:05SimplifiedBesselRadialBasisFunction/add_10/y:output:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/pow_6/yConst*
_output_shapes
: *
dtype0*
valueB 2      @З
)SimplifiedBesselRadialBasisFunction/pow_6Pow.SimplifiedBesselRadialBasisFunction/add_10:z:04SimplifiedBesselRadialBasisFunction/pow_6/y:output:0*
T0*
_output_shapes
: u
,SimplifiedBesselRadialBasisFunction/mul_10/xConst*
_output_shapes
: *
dtype0*
valueB 2      @И
*SimplifiedBesselRadialBasisFunction/mul_10Mul5SimplifiedBesselRadialBasisFunction/mul_10/x:output:0-SimplifiedBesselRadialBasisFunction/pow_6:z:0*
T0*
_output_shapes
: u
,SimplifiedBesselRadialBasisFunction/add_11/yConst*
_output_shapes
: *
dtype0*
valueB 2      №?Л
*SimplifiedBesselRadialBasisFunction/add_11AddV2.SimplifiedBesselRadialBasisFunction/mul_10:z:05SimplifiedBesselRadialBasisFunction/add_11/y:output:0*
T0*
_output_shapes
: И
-SimplifiedBesselRadialBasisFunction/truediv_6RealDiv-SimplifiedBesselRadialBasisFunction/mul_9:z:0.SimplifiedBesselRadialBasisFunction/add_11:z:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/Const_5Const*
_output_shapes
: *
dtype0*
valueB 2      №?Т
-SimplifiedBesselRadialBasisFunction/truediv_7RealDiv1SimplifiedBesselRadialBasisFunction/truediv_6:z:04SimplifiedBesselRadialBasisFunction/Const_3:output:0*
T0*
_output_shapes
: И
'SimplifiedBesselRadialBasisFunction/subSub4SimplifiedBesselRadialBasisFunction/Const_5:output:01SimplifiedBesselRadialBasisFunction/truediv_7:z:0*
T0*
_output_shapes
: 
*SimplifiedBesselRadialBasisFunction/Sqrt_2Sqrt+SimplifiedBesselRadialBasisFunction/sub:z:0*
T0*
_output_shapes
: x
/SimplifiedBesselRadialBasisFunction/truediv_8/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?У
-SimplifiedBesselRadialBasisFunction/truediv_8RealDiv8SimplifiedBesselRadialBasisFunction/truediv_8/x:output:0.SimplifiedBesselRadialBasisFunction/Sqrt_2:y:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/Const_6Const*
_output_shapes
: *
dtype0*
valueB 2      №ПН
)SimplifiedBesselRadialBasisFunction/Pow_7Pow4SimplifiedBesselRadialBasisFunction/Const_6:output:04SimplifiedBesselRadialBasisFunction/Const_4:output:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/Const_7Const*
_output_shapes
: *
dtype0*
valueB 2       @
*SimplifiedBesselRadialBasisFunction/Sqrt_3Sqrt4SimplifiedBesselRadialBasisFunction/Const_7:output:0*
T0*
_output_shapes
: Б
*SimplifiedBesselRadialBasisFunction/mul_11Mul-SimplifiedBesselRadialBasisFunction/Pow_7:z:0.SimplifiedBesselRadialBasisFunction/Sqrt_3:y:0*
T0*
_output_shapes
: Џ
*SimplifiedBesselRadialBasisFunction/mul_12Mul.SimplifiedBesselRadialBasisFunction/mul_11:z:0+simplifiedbesselradialbasisfunction_mul_1_y*
T0*
_output_shapes
: В
8SimplifiedBesselRadialBasisFunction/Pow_8/ReadVariableOpReadVariableOpAsimplifiedbesselradialbasisfunction_pow_1_readvariableop_resource*
_output_shapes
: *
dtype0t
+SimplifiedBesselRadialBasisFunction/Pow_8/yConst*
_output_shapes
: *
dtype0*
valueB 2      ј?Щ
)SimplifiedBesselRadialBasisFunction/Pow_8Pow@SimplifiedBesselRadialBasisFunction/Pow_8/ReadVariableOp:value:04SimplifiedBesselRadialBasisFunction/Pow_8/y:output:0*
T0*
_output_shapes
: И
-SimplifiedBesselRadialBasisFunction/truediv_9RealDiv.SimplifiedBesselRadialBasisFunction/mul_12:z:0-SimplifiedBesselRadialBasisFunction/Pow_8:z:0*
T0*
_output_shapes
: u
,SimplifiedBesselRadialBasisFunction/add_12/yConst*
_output_shapes
: *
dtype0*
valueB 2      №?С
*SimplifiedBesselRadialBasisFunction/add_12AddV24SimplifiedBesselRadialBasisFunction/Const_4:output:05SimplifiedBesselRadialBasisFunction/add_12/y:output:0*
T0*
_output_shapes
: Е
*SimplifiedBesselRadialBasisFunction/mul_13Mul1SimplifiedBesselRadialBasisFunction/truediv_9:z:0.SimplifiedBesselRadialBasisFunction/add_12:z:0*
T0*
_output_shapes
: u
,SimplifiedBesselRadialBasisFunction/add_13/yConst*
_output_shapes
: *
dtype0*
valueB 2       @С
*SimplifiedBesselRadialBasisFunction/add_13AddV24SimplifiedBesselRadialBasisFunction/Const_4:output:05SimplifiedBesselRadialBasisFunction/add_13/y:output:0*
T0*
_output_shapes
: В
*SimplifiedBesselRadialBasisFunction/mul_14Mul.SimplifiedBesselRadialBasisFunction/mul_13:z:0.SimplifiedBesselRadialBasisFunction/add_13:z:0*
T0*
_output_shapes
: u
,SimplifiedBesselRadialBasisFunction/add_14/yConst*
_output_shapes
: *
dtype0*
valueB 2      №?С
*SimplifiedBesselRadialBasisFunction/add_14AddV24SimplifiedBesselRadialBasisFunction/Const_4:output:05SimplifiedBesselRadialBasisFunction/add_14/y:output:0*
T0*
_output_shapes
: t
+SimplifiedBesselRadialBasisFunction/pow_9/yConst*
_output_shapes
: *
dtype0*
valueB 2       @З
)SimplifiedBesselRadialBasisFunction/pow_9Pow.SimplifiedBesselRadialBasisFunction/add_14:z:04SimplifiedBesselRadialBasisFunction/pow_9/y:output:0*
T0*
_output_shapes
: u
,SimplifiedBesselRadialBasisFunction/add_15/yConst*
_output_shapes
: *
dtype0*
valueB 2       @С
*SimplifiedBesselRadialBasisFunction/add_15AddV24SimplifiedBesselRadialBasisFunction/Const_4:output:05SimplifiedBesselRadialBasisFunction/add_15/y:output:0*
T0*
_output_shapes
: u
,SimplifiedBesselRadialBasisFunction/pow_10/yConst*
_output_shapes
: *
dtype0*
valueB 2       @Й
*SimplifiedBesselRadialBasisFunction/pow_10Pow.SimplifiedBesselRadialBasisFunction/add_15:z:05SimplifiedBesselRadialBasisFunction/pow_10/y:output:0*
T0*
_output_shapes
: Г
*SimplifiedBesselRadialBasisFunction/add_16AddV2-SimplifiedBesselRadialBasisFunction/pow_9:z:0.SimplifiedBesselRadialBasisFunction/pow_10:z:0*
T0*
_output_shapes
: 
*SimplifiedBesselRadialBasisFunction/Sqrt_4Sqrt.SimplifiedBesselRadialBasisFunction/add_16:z:0*
T0*
_output_shapes
: К
.SimplifiedBesselRadialBasisFunction/truediv_10RealDiv.SimplifiedBesselRadialBasisFunction/mul_14:z:0.SimplifiedBesselRadialBasisFunction/Sqrt_4:y:0*
T0*
_output_shapes
: u
,SimplifiedBesselRadialBasisFunction/add_17/yConst*
_output_shapes
: *
dtype0*
valueB 2      №?С
*SimplifiedBesselRadialBasisFunction/add_17AddV24SimplifiedBesselRadialBasisFunction/Const_4:output:05SimplifiedBesselRadialBasisFunction/add_17/y:output:0*
T0*
_output_shapes
: Ъ
*SimplifiedBesselRadialBasisFunction/mul_15Mul5SimplifiedBesselRadialBasisFunction/SelectV2:output:0.SimplifiedBesselRadialBasisFunction/add_17:z:0*
T0*'
_output_shapes
:џџџџџџџџџР
*SimplifiedBesselRadialBasisFunction/mul_16Mul.SimplifiedBesselRadialBasisFunction/mul_15:z:0+simplifiedbesselradialbasisfunction_mul_1_y*
T0*'
_output_shapes
:џџџџџџџџџЗ
=SimplifiedBesselRadialBasisFunction/truediv_11/ReadVariableOpReadVariableOpAsimplifiedbesselradialbasisfunction_pow_1_readvariableop_resource*
_output_shapes
: *
dtype0т
.SimplifiedBesselRadialBasisFunction/truediv_11RealDiv.SimplifiedBesselRadialBasisFunction/mul_16:z:0ESimplifiedBesselRadialBasisFunction/truediv_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџy
0SimplifiedBesselRadialBasisFunction/NotEqual_2/yConst*
_output_shapes
: *
dtype0*
valueB 2        ћ
.SimplifiedBesselRadialBasisFunction/NotEqual_2NotEqual2SimplifiedBesselRadialBasisFunction/truediv_11:z:09SimplifiedBesselRadialBasisFunction/NotEqual_2/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
incompatible_shape_error( 
)SimplifiedBesselRadialBasisFunction/Sin_2Sin2SimplifiedBesselRadialBasisFunction/truediv_11:z:0*
T0*'
_output_shapes
:џџџџџџџџџЮ
.SimplifiedBesselRadialBasisFunction/truediv_12RealDiv-SimplifiedBesselRadialBasisFunction/Sin_2:y:02SimplifiedBesselRadialBasisFunction/truediv_11:z:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
/SimplifiedBesselRadialBasisFunction/ones_like_2OnesLike2SimplifiedBesselRadialBasisFunction/truediv_11:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
.SimplifiedBesselRadialBasisFunction/SelectV2_3SelectV22SimplifiedBesselRadialBasisFunction/NotEqual_2:z:02SimplifiedBesselRadialBasisFunction/truediv_12:z:03SimplifiedBesselRadialBasisFunction/ones_like_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџu
,SimplifiedBesselRadialBasisFunction/add_18/yConst*
_output_shapes
: *
dtype0*
valueB 2       @С
*SimplifiedBesselRadialBasisFunction/add_18AddV24SimplifiedBesselRadialBasisFunction/Const_4:output:05SimplifiedBesselRadialBasisFunction/add_18/y:output:0*
T0*
_output_shapes
: Ъ
*SimplifiedBesselRadialBasisFunction/mul_17Mul5SimplifiedBesselRadialBasisFunction/SelectV2:output:0.SimplifiedBesselRadialBasisFunction/add_18:z:0*
T0*'
_output_shapes
:џџџџџџџџџР
*SimplifiedBesselRadialBasisFunction/mul_18Mul.SimplifiedBesselRadialBasisFunction/mul_17:z:0+simplifiedbesselradialbasisfunction_mul_1_y*
T0*'
_output_shapes
:џџџџџџџџџЗ
=SimplifiedBesselRadialBasisFunction/truediv_13/ReadVariableOpReadVariableOpAsimplifiedbesselradialbasisfunction_pow_1_readvariableop_resource*
_output_shapes
: *
dtype0т
.SimplifiedBesselRadialBasisFunction/truediv_13RealDiv.SimplifiedBesselRadialBasisFunction/mul_18:z:0ESimplifiedBesselRadialBasisFunction/truediv_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџy
0SimplifiedBesselRadialBasisFunction/NotEqual_3/yConst*
_output_shapes
: *
dtype0*
valueB 2        ћ
.SimplifiedBesselRadialBasisFunction/NotEqual_3NotEqual2SimplifiedBesselRadialBasisFunction/truediv_13:z:09SimplifiedBesselRadialBasisFunction/NotEqual_3/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
incompatible_shape_error( 
)SimplifiedBesselRadialBasisFunction/Sin_3Sin2SimplifiedBesselRadialBasisFunction/truediv_13:z:0*
T0*'
_output_shapes
:џџџџџџџџџЮ
.SimplifiedBesselRadialBasisFunction/truediv_14RealDiv-SimplifiedBesselRadialBasisFunction/Sin_3:y:02SimplifiedBesselRadialBasisFunction/truediv_13:z:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
/SimplifiedBesselRadialBasisFunction/ones_like_3OnesLike2SimplifiedBesselRadialBasisFunction/truediv_13:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
.SimplifiedBesselRadialBasisFunction/SelectV2_4SelectV22SimplifiedBesselRadialBasisFunction/NotEqual_3:z:02SimplifiedBesselRadialBasisFunction/truediv_14:z:03SimplifiedBesselRadialBasisFunction/ones_like_3:y:0*
T0*'
_output_shapes
:џџџџџџџџџз
*SimplifiedBesselRadialBasisFunction/add_19AddV27SimplifiedBesselRadialBasisFunction/SelectV2_3:output:07SimplifiedBesselRadialBasisFunction/SelectV2_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџЧ
*SimplifiedBesselRadialBasisFunction/mul_19Mul2SimplifiedBesselRadialBasisFunction/truediv_10:z:0.SimplifiedBesselRadialBasisFunction/add_19:z:0*
T0*'
_output_shapes
:џџџџџџџџџУ
.SimplifiedBesselRadialBasisFunction/truediv_15RealDiv1SimplifiedBesselRadialBasisFunction/truediv_6:z:04SimplifiedBesselRadialBasisFunction/Const_3:output:0*
T0*
_output_shapes
: 
*SimplifiedBesselRadialBasisFunction/Sqrt_5Sqrt2SimplifiedBesselRadialBasisFunction/truediv_15:z:0*
T0*
_output_shapes
: Т
*SimplifiedBesselRadialBasisFunction/mul_20Mul.SimplifiedBesselRadialBasisFunction/Sqrt_5:y:0-SimplifiedBesselRadialBasisFunction/mul_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџХ
*SimplifiedBesselRadialBasisFunction/add_20AddV2.SimplifiedBesselRadialBasisFunction/mul_19:z:0.SimplifiedBesselRadialBasisFunction/mul_20:z:0*
T0*'
_output_shapes
:џџџџџџџџџЦ
*SimplifiedBesselRadialBasisFunction/mul_21Mul1SimplifiedBesselRadialBasisFunction/truediv_8:z:0.SimplifiedBesselRadialBasisFunction/add_20:z:0*
T0*'
_output_shapes
:џџџџџџџџџЯ
)SimplifiedBesselRadialBasisFunction/stackPack-SimplifiedBesselRadialBasisFunction/mul_8:z:0.SimplifiedBesselRadialBasisFunction/mul_21:z:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ
2SimplifiedBesselRadialBasisFunction/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          с
-SimplifiedBesselRadialBasisFunction/transpose	Transpose2SimplifiedBesselRadialBasisFunction/stack:output:0;SimplifiedBesselRadialBasisFunction/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
7SimplifiedBesselRadialBasisFunction/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
9SimplifiedBesselRadialBasisFunction/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
9SimplifiedBesselRadialBasisFunction/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         З
1SimplifiedBesselRadialBasisFunction/strided_sliceStridedSlice1SimplifiedBesselRadialBasisFunction/transpose:y:0@SimplifiedBesselRadialBasisFunction/strided_slice/stack:output:0BSimplifiedBesselRadialBasisFunction/strided_slice/stack_1:output:0BSimplifiedBesselRadialBasisFunction/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskv
-SimplifiedBesselRadialBasisFunction/Greater/yConst*
_output_shapes
: *
dtype0*
valueB 2      @з
+SimplifiedBesselRadialBasisFunction/GreaterGreater5SimplifiedBesselRadialBasisFunction/SelectV2:output:06SimplifiedBesselRadialBasisFunction/Greater/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџЉ
.SimplifiedBesselRadialBasisFunction/zeros_like	ZerosLike:SimplifiedBesselRadialBasisFunction/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
.SimplifiedBesselRadialBasisFunction/SelectV2_5SelectV2/SimplifiedBesselRadialBasisFunction/Greater:z:02SimplifiedBesselRadialBasisFunction/zeros_like:y:0:SimplifiedBesselRadialBasisFunction/strided_slice:output:0*
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

:@Ъ
DenseLayer/einsum/EinsumEinsum7SimplifiedBesselRadialBasisFunction/SelectV2_5:output:0DenseLayer/mul:z:0*
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
:џџџџџџџџџ@щ
DenseLayer/IdentityN	IdentityNDenseLayer/mul_2:z:0!DenseLayer/einsum/Einsum:output:0DenseLayer/Cast:y:0*
T
2*,
_gradient_op_typeCustomGradient-140728*<
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
:џџџџџџџџџ@я
DenseLayer/IdentityN_1	IdentityNDenseLayer/mul_6:z:0#DenseLayer/einsum_1/Einsum:output:0DenseLayer/Cast_1:y:0*
T
2*,
_gradient_op_typeCustomGradient-140745*<
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
:џџџџџџџџџ@№
DenseLayer/IdentityN_2	IdentityNDenseLayer/mul_10:z:0#DenseLayer/einsum_2/Einsum:output:0DenseLayer/Cast_2:y:0*
T
2*,
_gradient_op_typeCustomGradient-140762*<
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
valueB:л
Y/strided_slice_3StridedSlicey_140819 Y/strided_slice_3/stack:output:0"Y/strided_slice_3/stack_1:output:0"Y/strided_slice_3/stack_2:output:0*
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
valueB:л
Y/strided_slice_4StridedSlicey_140826 Y/strided_slice_4/stack:output:0"Y/strided_slice_4/stack_1:output:0"Y/strided_slice_4/stack_2:output:0*
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
valueB:л
Y/strided_slice_5StridedSlicey_140819 Y/strided_slice_5/stack:output:0"Y/strided_slice_5/stack_1:output:0"Y/strided_slice_5/stack_2:output:0*
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
:џџџџџџџџџ	R
I/zeros/packed/0Const*
_output_shapes
: *
dtype0*
value	B :R
I/zeros/packed/2Const*
_output_shapes
: *
dtype0*
value	B :
I/zeros/packedPackI/zeros/packed/0:output:0batch_tot_natI/zeros/packed/2:output:0*
N*
T0*
_output_shapes
:V
I/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        v
I/zerosFillI/zeros/packed:output:0I/zeros/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ{
I/GatherV2/indicesConst*
_output_shapes
:*
dtype0	*5
value,B*	"                              Q
I/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Б

I/GatherV2GatherV2	A/mul:z:0I/GatherV2/indices:output:0I/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџ
I/GatherV2_1/ReadVariableOpReadVariableOp$i_gatherv2_1_readvariableop_resource*&
_output_shapes
:*
dtype0\
I/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџУ
I/GatherV2_1GatherV2#I/GatherV2_1/ReadVariableOp:value:0i_gatherv2_1_indicesI/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*&
_output_shapes
:S
I/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : Е
I/GatherV2_2GatherV2I/GatherV2_1:output:0atomic_mu_iI/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:џџџџџџџџџ 
I/ein_A/EinsumEinsumI/GatherV2_2:output:0I/GatherV2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationaknw,anw->wakd
I/mulMulI/ein_A/Einsum:output:0i_mul_y*
T0*+
_output_shapes
:џџџџџџџџџЅ
I/TensorScatterAddTensorScatterAddI/zeros:output:0i_tensorscatteradd_indices	I/mul:z:0*
Tindices0*
T0*+
_output_shapes
:џџџџџџџџџe
I/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
I/transpose	TransposeI/TensorScatterAdd:output:0I/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
E/zeros/packed/0Const*
_output_shapes
: *
dtype0*
value	B :R
E/zeros/packed/2Const*
_output_shapes
: *
dtype0*
value	B :
E/zeros/packedPackE/zeros/packed/0:output:0batch_tot_natE/zeros/packed/2:output:0*
N*
T0*
_output_shapes
:V
E/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        v
E/zerosFillE/zeros/packed:output:0E/zeros/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ\
E/GatherV2/indicesConst*
_output_shapes
:*
dtype0	*
valueB	R Q
E/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Б

E/GatherV2GatherV2	A/mul:z:0E/GatherV2/indices:output:0E/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџ
E/GatherV2_1/ReadVariableOpReadVariableOp$e_gatherv2_1_readvariableop_resource*&
_output_shapes
:*
dtype0\
E/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџУ
E/GatherV2_1GatherV2#E/GatherV2_1/ReadVariableOp:value:0e_gatherv2_1_indicesE/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*&
_output_shapes
:S
E/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : Е
E/GatherV2_2GatherV2E/GatherV2_1:output:0atomic_mu_iE/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:џџџџџџџџџ 
E/ein_A/EinsumEinsumE/GatherV2_2:output:0E/GatherV2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationaknw,anw->wakd
E/mulMulE/ein_A/Einsum:output:0e_mul_y*
T0*+
_output_shapes
:џџџџџџџџџЅ
E/TensorScatterAddTensorScatterAddE/zeros:output:0e_tensorscatteradd_indices	E/mul:z:0*
Tindices0*
T0*+
_output_shapes
:џџџџџџџџџe
E/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
E/transpose	TransposeE/TensorScatterAdd:output:0E/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџT
rho/zeros/packed/0Const*
_output_shapes
: *
dtype0*
value	B :T
rho/zeros/packed/2Const*
_output_shapes
: *
dtype0*
value	B :
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
:џџџџџџџџџ^
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
:*
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
:U
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
:џџџџџџџџџІ
rho/ein_A/EinsumEinsumrho/GatherV2_2:output:0rho/GatherV2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationaknw,anw->wakj
rho/mulMulrho/ein_A/Einsum:output:0	rho_mul_y*
T0*+
_output_shapes
:џџџџџџџџџ­
rho/TensorScatterAddTensorScatterAddrho/zeros:output:0rho_tensorscatteradd_indicesrho/mul:z:0*
Tindices0*
T0*+
_output_shapes
:џџџџџџџџџg
rho/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
rho/transpose	Transposerho/TensorScatterAdd:output:0rho/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
YI/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ѓ
YI/GatherV2GatherV2I/transpose:y:0ind_jYI/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:џџџџџџџџџT
YI/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :М
YI/GatherV2_1GatherV2YI/GatherV2:output:0yi_gatherv2_1_indicesYI/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:џџџџџџџџџ%
YI/einsum/EinsumEinsumR/GatherV2:output:0Y/mul_35:z:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ	*
equationjnl,jl->jnlT
YI/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B :С
YI/GatherV2_2GatherV2YI/einsum/Einsum:output:0yi_gatherv2_2_indicesYI/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:џџџџџџџџџ%s
YI/mulMulYI/GatherV2_2:output:0YI/GatherV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ%]
YI/mul_1Mul
YI/mul:z:0
yi_mul_1_y*
T0*+
_output_shapes
:џџџџџџџџџ%h
YI/trans_201YI/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
YI/trans_201YI	TransposeYI/mul_1:z:0YI/trans_201YI/perm:output:0*
T0*+
_output_shapes
:%џџџџџџџџџБ
YI/sum_cg_YIUnsortedSegmentSumYI/trans_201YI:y:0yi_sum_cg_yi_segment_idsyi_sum_cg_yi_num_segments*
Tindices0*
T0*+
_output_shapes
:џџџџџџџџџh
YI/trans_120YI/permConst*
_output_shapes
:*
dtype0*!
valueB"          
YI/trans_120YI	TransposeYI/sum_cg_YI:output:0YI/trans_120YI/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
YI/sum_nei_YIUnsortedSegmentSumYI/trans_120YI:y:0ind_ibatch_tot_nat*
Tindices0*
T0*+
_output_shapes
:џџџџџџџџџi
YI/mul_2MulYI/sum_nei_YI:output:0
yi_mul_2_y*
T0*+
_output_shapes
:џџџџџџџџџR
B/zeros/packed/0Const*
_output_shapes
: *
dtype0*
value	B :	R
B/zeros/packed/2Const*
_output_shapes
: *
dtype0*
value	B :
B/zeros/packedPackB/zeros/packed/0:output:0batch_tot_natB/zeros/packed/2:output:0*
N*
T0*
_output_shapes
:V
B/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        v
B/zerosFillB/zeros/packed:output:0B/zeros/Const:output:0*
T0*+
_output_shapes
:	џџџџџџџџџ
B/GatherV2/indicesConst*
_output_shapes
:*
dtype0	*Р
valueЖBГ	"Ј                                                 	       
                                                                                            Q
B/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Д

B/GatherV2GatherV2YI/mul_2:z:0B/GatherV2/indices:output:0B/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџ
B/GatherV2_1/ReadVariableOpReadVariableOp$b_gatherv2_1_readvariableop_resource*"
_output_shapes
:*
dtype0\
B/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџП
B/GatherV2_1GatherV2#B/GatherV2_1/ReadVariableOp:value:0b_gatherv2_1_indicesB/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*"
_output_shapes
: 
B/ein_YI/EinsumEinsumB/GatherV2_1:output:0B/GatherV2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationknw,anw->wake
B/mulMulB/ein_YI/Einsum:output:0b_mul_y*
T0*+
_output_shapes
:џџџџџџџџџЅ
B/TensorScatterAddTensorScatterAddB/zeros:output:0b_tensorscatteradd_indices	B/mul:z:0*
Tindices0*
T0*+
_output_shapes
:	џџџџџџџџџe
B/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
B/transpose	TransposeB/TensorScatterAdd:output:0B/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ	S
E2/zeros/packed/0Const*
_output_shapes
: *
dtype0*
value	B :S
E2/zeros/packed/2Const*
_output_shapes
: *
dtype0*
value	B :
E2/zeros/packedPackE2/zeros/packed/0:output:0batch_tot_natE2/zeros/packed/2:output:0*
N*
T0*
_output_shapes
:W
E2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        y
E2/zerosFillE2/zeros/packed:output:0E2/zeros/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ]
E2/GatherV2/indicesConst*
_output_shapes
:*
dtype0	*
valueB	RR
E2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :К
E2/GatherV2GatherV2B/transpose:y:0E2/GatherV2/indices:output:0E2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџ
E2/GatherV2_1/ReadVariableOpReadVariableOp%e2_gatherv2_1_readvariableop_resource*&
_output_shapes
:*
dtype0]
E2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЧ
E2/GatherV2_1GatherV2$E2/GatherV2_1/ReadVariableOp:value:0e2_gatherv2_1_indicesE2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*&
_output_shapes
:T
E2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : И
E2/GatherV2_2GatherV2E2/GatherV2_1:output:0atomic_mu_iE2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:џџџџџџџџџЃ
E2/ein_B/EinsumEinsumE2/GatherV2_2:output:0E2/GatherV2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationaknw,anw->wakg
E2/mulMulE2/ein_B/Einsum:output:0e2_mul_y*
T0*+
_output_shapes
:џџџџџџџџџЉ
E2/TensorScatterAddTensorScatterAddE2/zeros:output:0e2_tensorscatteradd_indices
E2/mul:z:0*
Tindices0*
T0*+
_output_shapes
:џџџџџџџџџf
E2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
E2/transpose	TransposeE2/TensorScatterAdd:output:0E2/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџU
rho2/zeros/packed/0Const*
_output_shapes
: *
dtype0*
value	B :U
rho2/zeros/packed/2Const*
_output_shapes
: *
dtype0*
value	B :
rho2/zeros/packedPackrho2/zeros/packed/0:output:0batch_tot_natrho2/zeros/packed/2:output:0*
N*
T0*
_output_shapes
:Y
rho2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        

rho2/zerosFillrho2/zeros/packed:output:0rho2/zeros/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ_
rho2/GatherV2/indicesConst*
_output_shapes
:*
dtype0	*
valueB	RT
rho2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Р
rho2/GatherV2GatherV2B/transpose:y:0rho2/GatherV2/indices:output:0rho2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџ
rho2/GatherV2_1/ReadVariableOpReadVariableOp'rho2_gatherv2_1_readvariableop_resource*&
_output_shapes
:*
dtype0_
rho2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЯ
rho2/GatherV2_1GatherV2&rho2/GatherV2_1/ReadVariableOp:value:0rho2_gatherv2_1_indicesrho2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*&
_output_shapes
:V
rho2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : О
rho2/GatherV2_2GatherV2rho2/GatherV2_1:output:0atomic_mu_irho2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:џџџџџџџџџЉ
rho2/ein_B/EinsumEinsumrho2/GatherV2_2:output:0rho2/GatherV2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationaknw,anw->wakm
rho2/mulMulrho2/ein_B/Einsum:output:0
rho2_mul_y*
T0*+
_output_shapes
:џџџџџџџџџБ
rho2/TensorScatterAddTensorScatterAddrho2/zeros:output:0rho2_tensorscatteradd_indicesrho2/mul:z:0*
Tindices0*
T0*+
_output_shapes
:џџџџџџџџџh
rho2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
rho2/transpose	Transposerho2/TensorScatterAdd:output:0rho2/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџh
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
valueB"         
strided_sliceStridedSliceE/transpose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskN
add/xConst*
_output_shapes
: *
dtype0*
valueB 2        f
addAddV2add/x:output:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
strided_slice_1StridedSliceE2/transpose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskc
add_1AddV2add:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_2AddV2add_2_x	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџj
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
strided_slice_2StridedSlicerho/transpose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskP
add_3/xConst*
_output_shapes
: *
dtype0*
valueB 2        l
add_3AddV2add_3/x:output:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
strided_slice_3StridedSlicerho2/transpose:y:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maske
add_4AddV2	add_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
DenseLayer/ReadVariableOp_4ReadVariableOp$denselayer_readvariableop_4_resource*
_output_shapes

: *
dtype0{
DenseLayer/mul_13Mul#DenseLayer/ReadVariableOp_4:value:0denselayer_mul_13_y*
T0*
_output_shapes

: Ё
DenseLayer/einsum_4/EinsumEinsum	add_4:z:0DenseLayer/mul_13:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ *
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
:џџџџџџџџџ h
DenseLayer/Sigmoid_3SigmoidDenseLayer/mul_14:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 
DenseLayer/mul_15Mul#DenseLayer/einsum_4/Einsum:output:0DenseLayer/Sigmoid_3:y:0*
T0*'
_output_shapes
:џџџџџџџџџ j
DenseLayer/Identity_3IdentityDenseLayer/mul_15:z:0*
T0*'
_output_shapes
:џџџџџџџџџ №
DenseLayer/IdentityN_3	IdentityNDenseLayer/mul_15:z:0#DenseLayer/einsum_4/Einsum:output:0DenseLayer/Cast_3:y:0*
T
2*,
_gradient_op_typeCustomGradient-141224*<
_output_shapes*
(:џџџџџџџџџ :џџџџџџџџџ : \
DenseLayer/mul_16/yConst*
_output_shapes
: *
dtype0*
valueB 2ЦмЕ|ањ?
DenseLayer/mul_16MulDenseLayer/IdentityN_3:output:0DenseLayer/mul_16/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
DenseLayer/ReadVariableOp_5ReadVariableOp$denselayer_readvariableop_5_resource*
_output_shapes

: *
dtype0{
DenseLayer/mul_17Mul#DenseLayer/ReadVariableOp_5:value:0denselayer_mul_17_y*
T0*
_output_shapes

: ­
DenseLayer/einsum_5/EinsumEinsumDenseLayer/mul_16:z:0DenseLayer/mul_17:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ*
equation...k,...kn->...np
add_5AddV2	add_2:z:0#DenseLayer/einsum_5/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџN
mulMul	add_5:z:0mul_y*
T0*'
_output_shapes
:џџџџџџџџџ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   e
ReshapeReshapemul:z:0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]
ones_like/ShapeShapeReshape:output:0*
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
:џџџџџџџџџX
gradient_tape/ShapeShapemul:z:0*
T0*
_output_shapes
::эЯ
gradient_tape/ReshapeReshapeones_like:output:0gradient_tape/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџu
gradient_tape/mul/MulMulgradient_tape/Reshape:output:0mul_y*
T0*'
_output_shapes
:џџџџџџџџџ`
gradient_tape/add_5/ShapeShape	add_2:z:0*
T0*
_output_shapes
::эЯ|
gradient_tape/add_5/Shape_1Shape#DenseLayer/einsum_5/Einsum:output:0*
T0*
_output_shapes
::эЯР
)gradient_tape/add_5/BroadcastGradientArgsBroadcastGradientArgs"gradient_tape/add_5/Shape:output:0$gradient_tape/add_5/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЕ
gradient_tape/add_5/SumSumgradient_tape/mul/Mul:z:0.gradient_tape/add_5/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
gradient_tape/add_5/ReshapeReshape gradient_tape/add_5/Sum:output:0"gradient_tape/add_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЗ
gradient_tape/add_5/Sum_1Sumgradient_tape/mul/Mul:z:0.gradient_tape/add_5/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(Є
gradient_tape/add_5/Reshape_1Reshape"gradient_tape/add_5/Sum_1:output:0$gradient_tape/add_5/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\
gradient_tape/add_2/ShapeConst*
_output_shapes
: *
dtype0*
valueB b
gradient_tape/add_2/Shape_1Shape	add_1:z:0*
T0*
_output_shapes
::эЯz
'gradient_tape/DenseLayer/einsum_5/ShapeShapeDenseLayer/mul_16:z:0*
T0*
_output_shapes
::эЯz
)gradient_tape/DenseLayer/einsum_5/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"       Ь
(gradient_tape/DenseLayer/einsum_5/EinsumEinsum&gradient_tape/add_5/Reshape_1:output:0DenseLayer/mul_17:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ *
equation...n,...kn->...kв
*gradient_tape/DenseLayer/einsum_5/Einsum_1Einsum&gradient_tape/add_5/Reshape_1:output:0DenseLayer/mul_16:z:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ *
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
:џџџџџџџџџ k
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

: ^
gradient_tape/add_1/ShapeShapeadd:z:0*
T0*
_output_shapes
::эЯq
gradient_tape/add_1/Shape_1Shapestrided_slice_1:output:0*
T0*
_output_shapes
::эЯР
)gradient_tape/add_1/BroadcastGradientArgsBroadcastGradientArgs"gradient_tape/add_1/Shape:output:0$gradient_tape/add_1/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџР
gradient_tape/add_1/SumSum$gradient_tape/add_5/Reshape:output:0.gradient_tape/add_1/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
gradient_tape/add_1/ReshapeReshape gradient_tape/add_1/Sum:output:0"gradient_tape/add_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџТ
gradient_tape/add_1/Sum_1Sum$gradient_tape/add_5/Reshape:output:0.gradient_tape/add_1/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(Є
gradient_tape/add_1/Reshape_1Reshape"gradient_tape/add_1/Sum_1:output:0$gradient_tape/add_1/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџж
#gradient_tape/DenseLayer/mul_16/MulMul2gradient_tape/DenseLayer/einsum_5/Reshape:output:0DenseLayer/mul_16/y:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ 
#gradient_tape/DenseLayer/mul_17/MulMul4gradient_tape/DenseLayer/einsum_5/Reshape_1:output:0denselayer_mul_17_y*
T0*
_output_shapes

: Z
gradient_tape/add/ShapeConst*
_output_shapes
: *
dtype0*
valueB m
gradient_tape/add/Shape_1Shapestrided_slice:output:0*
T0*
_output_shapes
::эЯq
#gradient_tape/strided_slice_1/ShapeShapeE2/transpose:y:0*
T0*
_output_shapes
::эЯ
4gradient_tape/strided_slice_1/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*!
valueB"            
2gradient_tape/strided_slice_1/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*!
valueB"           
6gradient_tape/strided_slice_1/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*!
valueB"         в
.gradient_tape/strided_slice_1/StridedSliceGradStridedSliceGrad,gradient_tape/strided_slice_1/Shape:output:0=gradient_tape/strided_slice_1/StridedSliceGrad/begin:output:0;gradient_tape/strided_slice_1/StridedSliceGrad/end:output:0?gradient_tape/strided_slice_1/StridedSliceGrad/strides:output:0&gradient_tape/add_1/Reshape_1:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskj

zeros_like	ZerosLikeDenseLayer/IdentityN_3:output:1*
T0*'
_output_shapes
:џџџџџџџџџ N
zerosConst*
_output_shapes
: *
dtype0*
valueB 2         
mul_1MulDenseLayer/Cast_3:y:0#DenseLayer/einsum_4/Einsum:output:0$^gradient_tape/DenseLayer/mul_16/Mul*
T0*'
_output_shapes
:џџџџџџџџџ O
SigmoidSigmoid	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ z
mul_2MulDenseLayer/Cast_3:y:0#DenseLayer/einsum_4/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ N
sub/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ R
mul_3Mul	mul_2:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ P
add_6/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?]
add_6AddV2add_6/x:output:0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ V
mul_4MulSigmoid:y:0	add_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ g
SquareSquare#DenseLayer/einsum_4/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ s
mul_5Mul'gradient_tape/DenseLayer/mul_16/Mul:z:0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
mul_6Mul	mul_5:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ P
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?]
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ T
mul_7Mul	mul_6:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_7:z:0Const:output:0*
T0*
_output_shapes
: r
mul_8Mul'gradient_tape/DenseLayer/mul_16/Mul:z:0	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
!gradient_tape/strided_slice/ShapeShapeE/transpose:y:0*
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
valueB"         Ц
,gradient_tape/strided_slice/StridedSliceGradStridedSliceGrad*gradient_tape/strided_slice/Shape:output:0;gradient_tape/strided_slice/StridedSliceGrad/begin:output:09gradient_tape/strided_slice/StridedSliceGrad/end:output:0=gradient_tape/strided_slice/StridedSliceGrad/strides:output:0$gradient_tape/add_1/Reshape:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_masky
,gradient_tape/E2/transpose/InvertPermutationInvertPermutationE2/transpose/perm:output:0*
_output_shapes
:в
$gradient_tape/E2/transpose/transpose	Transpose7gradient_tape/strided_slice_1/StridedSliceGrad:output:00gradient_tape/E2/transpose/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџw
+gradient_tape/E/transpose/InvertPermutationInvertPermutationE/transpose/perm:output:0*
_output_shapes
:Ю
#gradient_tape/E/transpose/transpose	Transpose5gradient_tape/strided_slice/StridedSliceGrad:output:0/gradient_tape/E/transpose/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџИ
gradient_tape/E2/GatherNdGatherNd(gradient_tape/E2/transpose/transpose:y:0e2_tensorscatteradd_indices*
Tindices0*
Tparams0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/E2/IdentityIdentity(gradient_tape/E2/transpose/transpose:y:0*
T0*+
_output_shapes
:џџџџџџџџџЕ
gradient_tape/E/GatherNdGatherNd'gradient_tape/E/transpose/transpose:y:0e_tensorscatteradd_indices*
Tindices0*
Tparams0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/E/IdentityIdentity'gradient_tape/E/transpose/transpose:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/E2/mul/MulMul"gradient_tape/E2/GatherNd:output:0e2_mul_y*
T0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/E/mul/MulMul!gradient_tape/E/GatherNd:output:0e_mul_y*
T0*+
_output_shapes
:џџџџџџџџџp
gradient_tape/E2/ein_B/ShapeShapeE2/GatherV2_2:output:0*
T0*
_output_shapes
::эЯp
gradient_tape/E2/ein_B/Shape_1ShapeE2/GatherV2:output:0*
T0*
_output_shapes
::эЯЛ
gradient_tape/E2/ein_B/EinsumEinsumgradient_tape/E2/mul/Mul:z:0E2/GatherV2:output:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ*
equationwak,anw->aknwЛ
gradient_tape/E2/ein_B/Einsum_1Einsumgradient_tape/E2/mul/Mul:z:0E2/GatherV2_2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationwak,aknw->anwn
'gradient_tape/DenseLayer/einsum_4/ShapeShape	add_4:z:0*
T0*
_output_shapes
::эЯz
)gradient_tape/DenseLayer/einsum_4/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"       Џ
(gradient_tape/DenseLayer/einsum_4/EinsumEinsum	mul_8:z:0DenseLayer/mul_13:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ*
equation...n,...kn->...kЉ
*gradient_tape/DenseLayer/einsum_4/Einsum_1Einsum	mul_8:z:0	add_4:z:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ *
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

: n
gradient_tape/E/ein_A/ShapeShapeE/GatherV2_2:output:0*
T0*
_output_shapes
::эЯn
gradient_tape/E/ein_A/Shape_1ShapeE/GatherV2:output:0*
T0*
_output_shapes
::эЯИ
gradient_tape/E/ein_A/EinsumEinsumgradient_tape/E/mul/Mul:z:0E/GatherV2:output:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ*
equationwak,anw->aknwИ
gradient_tape/E/ein_A/Einsum_1Einsumgradient_tape/E/mul/Mul:z:0E/GatherV2_2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationwak,aknw->anwЁ
gradient_tape/E2/ShapeConst* 
_class
loc:@E2/GatherV2_1*
_output_shapes
:*
dtype0	*5
value,B*	"                             
gradient_tape/E2/CastCastgradient_tape/E2/Shape:output:0*

DstT0*

SrcT0	* 
_class
loc:@E2/GatherV2_1*
_output_shapes
:K
gradient_tape/E2/SizeSizeatomic_mu_i*
T0*
_output_shapes
: a
gradient_tape/E2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E2/ExpandDims
ExpandDimsgradient_tape/E2/Size:output:0(gradient_tape/E2/ExpandDims/dim:output:0*
T0*
_output_shapes
:n
$gradient_tape/E2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&gradient_tape/E2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&gradient_tape/E2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gradient_tape/E2/strided_sliceStridedSlicegradient_tape/E2/Cast:y:0-gradient_tape/E2/strided_slice/stack:output:0/gradient_tape/E2/strided_slice/stack_1:output:0/gradient_tape/E2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask^
gradient_tape/E2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
gradient_tape/E2/concatConcatV2$gradient_tape/E2/ExpandDims:output:0'gradient_tape/E2/strided_slice:output:0%gradient_tape/E2/concat/axis:output:0*
N*
T0*
_output_shapes
:Ї
gradient_tape/E2/ReshapeReshape&gradient_tape/E2/ein_B/Einsum:output:0 gradient_tape/E2/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
gradient_tape/E2/Reshape_1Reshapeatomic_mu_i$gradient_tape/E2/ExpandDims:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/E2/Shape_1ShapeB/transpose:y:0*
T0*
_class
loc:@B/transpose*
_output_shapes
:*
out_type0	:эа
gradient_tape/E2/Cast_1Cast!gradient_tape/E2/Shape_1:output:0*

DstT0*

SrcT0	*
_class
loc:@B/transpose*
_output_shapes
:Y
gradient_tape/E2/Size_1Const*
_output_shapes
: *
dtype0*
value	B :c
!gradient_tape/E2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E2/ExpandDims_1
ExpandDims gradient_tape/E2/Size_1:output:0*gradient_tape/E2/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:X
gradient_tape/E2/ConstConst*
_output_shapes
: *
dtype0*
value	B : Z
gradient_tape/E2/Const_1Const*
_output_shapes
: *
dtype0*
value	B :}
&gradient_tape/E2/strided_slice_1/stackPackgradient_tape/E2/Const:output:0*
N*
T0*
_output_shapes
:y
(gradient_tape/E2/strided_slice_1/stack_1PackE2/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:
(gradient_tape/E2/strided_slice_1/stack_2Pack!gradient_tape/E2/Const_1:output:0*
N*
T0*
_output_shapes
:Ј
 gradient_tape/E2/strided_slice_1StridedSlicegradient_tape/E2/Cast_1:y:0/gradient_tape/E2/strided_slice_1/stack:output:01gradient_tape/E2/strided_slice_1/stack_1:output:01gradient_tape/E2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskZ
gradient_tape/E2/Const_2Const*
_output_shapes
: *
dtype0*
value	B : Z
gradient_tape/E2/Const_3Const*
_output_shapes
: *
dtype0*
value	B :w
&gradient_tape/E2/strided_slice_2/stackPackE2/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:
(gradient_tape/E2/strided_slice_2/stack_1Pack!gradient_tape/E2/Const_2:output:0*
N*
T0*
_output_shapes
:
(gradient_tape/E2/strided_slice_2/stack_2Pack!gradient_tape/E2/Const_3:output:0*
N*
T0*
_output_shapes
:І
 gradient_tape/E2/strided_slice_2StridedSlicegradient_tape/E2/Cast_1:y:0/gradient_tape/E2/strided_slice_2/stack:output:01gradient_tape/E2/strided_slice_2/stack_1:output:01gradient_tape/E2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskp
&gradient_tape/E2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(gradient_tape/E2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(gradient_tape/E2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:В
 gradient_tape/E2/strided_slice_3StridedSlice)gradient_tape/E2/strided_slice_2:output:0/gradient_tape/E2/strided_slice_3/stack:output:01gradient_tape/E2/strided_slice_3/stack_1:output:01gradient_tape/E2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masku
"gradient_tape/E2/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ`
gradient_tape/E2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
gradient_tape/E2/concat_1ConcatV2)gradient_tape/E2/strided_slice_1:output:0+gradient_tape/E2/concat_1/values_1:output:0)gradient_tape/E2/strided_slice_3:output:0'gradient_tape/E2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Y
gradient_tape/E2/Size_2Const*
_output_shapes
: *
dtype0*
value	B :Y
gradient_tape/E2/Size_3Const*
_output_shapes
: *
dtype0*
value	B :^
gradient_tape/E2/range/startConst*
_output_shapes
: *
dtype0*
value	B : ^
gradient_tape/E2/range/limitConst*
_output_shapes
: *
dtype0*
value	B : ^
gradient_tape/E2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ў
gradient_tape/E2/rangeRange%gradient_tape/E2/range/start:output:0%gradient_tape/E2/range/limit:output:0%gradient_tape/E2/range/delta:output:0*
_output_shapes
: `
gradient_tape/E2/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : `
gradient_tape/E2/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :Б
gradient_tape/E2/range_1Range'gradient_tape/E2/range_1/start:output:0 gradient_tape/E2/Size_3:output:0'gradient_tape/E2/range_1/delta:output:0*
_output_shapes
:X
gradient_tape/E2/add/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/E2/addAddV2 gradient_tape/E2/Size_3:output:0gradient_tape/E2/add/y:output:0*
T0*
_output_shapes
: `
gradient_tape/E2/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E2/range_2Rangegradient_tape/E2/add:z:0 gradient_tape/E2/Size_2:output:0'gradient_tape/E2/range_2/delta:output:0*
_output_shapes
: В
gradient_tape/E2/Reshape_2Reshape(gradient_tape/E2/ein_B/Einsum_1:output:0"gradient_tape/E2/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџz
"gradient_tape/E2/concat_2/values_1Pack gradient_tape/E2/Size_3:output:0*
N*
T0*
_output_shapes
:`
gradient_tape/E2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E2/concat_2ConcatV2gradient_tape/E2/range:output:0+gradient_tape/E2/concat_2/values_1:output:0!gradient_tape/E2/range_1:output:0!gradient_tape/E2/range_2:output:0'gradient_tape/E2/concat_2/axis:output:0*
N*
T0*
_output_shapes
:Џ
gradient_tape/E2/transpose	Transpose#gradient_tape/E2/Reshape_2:output:0"gradient_tape/E2/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ`
gradient_tape/E2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : з
gradient_tape/E2/GatherV2GatherV2gradient_tape/E2/Cast_1:y:0"gradient_tape/E2/concat_2:output:0'gradient_tape/E2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Z
gradient_tape/E2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :~
gradient_tape/E2/add_1AddV2E2/GatherV2/axis:output:0!gradient_tape/E2/add_1/y:output:0*
T0*
_output_shapes
: w
&gradient_tape/E2/strided_slice_4/stackPackE2/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:z
(gradient_tape/E2/strided_slice_4/stack_1Packgradient_tape/E2/add_1:z:0*
N*
T0*
_output_shapes
:r
(gradient_tape/E2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
 gradient_tape/E2/strided_slice_4StridedSlicegradient_tape/E2/Cast_1:y:0/gradient_tape/E2/strided_slice_4/stack:output:01gradient_tape/E2/strided_slice_4/stack_1:output:01gradient_tape/E2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
gradient_tape/E2/Size_4Const*
_output_shapes
: *
dtype0*
value	B :c
!gradient_tape/E2/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E2/ExpandDims_2
ExpandDims gradient_tape/E2/Size_4:output:0*gradient_tape/E2/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:
gradient_tape/E2/Reshape_3ReshapeE2/GatherV2/indices:output:0&gradient_tape/E2/ExpandDims_2:output:0*
T0	*
_output_shapes
:ј
#gradient_tape/E2/UnsortedSegmentSumUnsortedSegmentSumgradient_tape/E2/transpose:y:0#gradient_tape/E2/Reshape_3:output:0)gradient_tape/E2/strided_slice_4:output:0*
Tindices0	*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџZ
gradient_tape/E2/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/E2/add_2AddV2!gradient_tape/E2/range_1:output:0!gradient_tape/E2/add_2/y:output:0*
T0*
_output_shapes
:l
"gradient_tape/E2/concat_3/values_2Const*
_output_shapes
:*
dtype0*
valueB: `
gradient_tape/E2/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E2/concat_3ConcatV2gradient_tape/E2/range:output:0gradient_tape/E2/add_2:z:0+gradient_tape/E2/concat_3/values_2:output:0!gradient_tape/E2/range_2:output:0'gradient_tape/E2/concat_3/axis:output:0*
N*
T0*
_output_shapes
:К
gradient_tape/E2/transpose_1	Transpose,gradient_tape/E2/UnsortedSegmentSum:output:0"gradient_tape/E2/concat_3:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ`
gradient_tape/add_4/ShapeShape	add_3:z:0*
T0*
_output_shapes
::эЯq
gradient_tape/add_4/Shape_1Shapestrided_slice_3:output:0*
T0*
_output_shapes
::эЯР
)gradient_tape/add_4/BroadcastGradientArgsBroadcastGradientArgs"gradient_tape/add_4/Shape:output:0$gradient_tape/add_4/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЮ
gradient_tape/add_4/SumSum2gradient_tape/DenseLayer/einsum_4/Reshape:output:0.gradient_tape/add_4/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
gradient_tape/add_4/ReshapeReshape gradient_tape/add_4/Sum:output:0"gradient_tape/add_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџа
gradient_tape/add_4/Sum_1Sum2gradient_tape/DenseLayer/einsum_4/Reshape:output:0.gradient_tape/add_4/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(Є
gradient_tape/add_4/Reshape_1Reshape"gradient_tape/add_4/Sum_1:output:0$gradient_tape/add_4/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
#gradient_tape/DenseLayer/mul_13/MulMul4gradient_tape/DenseLayer/einsum_4/Reshape_1:output:0denselayer_mul_13_y*
T0*
_output_shapes

: 
gradient_tape/E/ShapeConst*
_class
loc:@E/GatherV2_1*
_output_shapes
:*
dtype0	*5
value,B*	"                             
gradient_tape/E/CastCastgradient_tape/E/Shape:output:0*

DstT0*

SrcT0	*
_class
loc:@E/GatherV2_1*
_output_shapes
:J
gradient_tape/E/SizeSizeatomic_mu_i*
T0*
_output_shapes
: `
gradient_tape/E/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E/ExpandDims
ExpandDimsgradient_tape/E/Size:output:0'gradient_tape/E/ExpandDims/dim:output:0*
T0*
_output_shapes
:m
#gradient_tape/E/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:o
%gradient_tape/E/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%gradient_tape/E/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gradient_tape/E/strided_sliceStridedSlicegradient_tape/E/Cast:y:0,gradient_tape/E/strided_slice/stack:output:0.gradient_tape/E/strided_slice/stack_1:output:0.gradient_tape/E/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask]
gradient_tape/E/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : У
gradient_tape/E/concatConcatV2#gradient_tape/E/ExpandDims:output:0&gradient_tape/E/strided_slice:output:0$gradient_tape/E/concat/axis:output:0*
N*
T0*
_output_shapes
:Є
gradient_tape/E/ReshapeReshape%gradient_tape/E/ein_A/Einsum:output:0gradient_tape/E/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
gradient_tape/E/Reshape_1Reshapeatomic_mu_i#gradient_tape/E/ExpandDims:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/E/Shape_1Shape	A/mul:z:0*
T0*
_class

loc:@A/mul*
_output_shapes
:*
out_type0	:эа
gradient_tape/E/Cast_1Cast gradient_tape/E/Shape_1:output:0*

DstT0*

SrcT0	*
_class

loc:@A/mul*
_output_shapes
:X
gradient_tape/E/Size_1Const*
_output_shapes
: *
dtype0*
value	B :b
 gradient_tape/E/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E/ExpandDims_1
ExpandDimsgradient_tape/E/Size_1:output:0)gradient_tape/E/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:W
gradient_tape/E/ConstConst*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/E/Const_1Const*
_output_shapes
: *
dtype0*
value	B :{
%gradient_tape/E/strided_slice_1/stackPackgradient_tape/E/Const:output:0*
N*
T0*
_output_shapes
:w
'gradient_tape/E/strided_slice_1/stack_1PackE/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/E/strided_slice_1/stack_2Pack gradient_tape/E/Const_1:output:0*
N*
T0*
_output_shapes
:Ѓ
gradient_tape/E/strided_slice_1StridedSlicegradient_tape/E/Cast_1:y:0.gradient_tape/E/strided_slice_1/stack:output:00gradient_tape/E/strided_slice_1/stack_1:output:00gradient_tape/E/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskY
gradient_tape/E/Const_2Const*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/E/Const_3Const*
_output_shapes
: *
dtype0*
value	B :u
%gradient_tape/E/strided_slice_2/stackPackE/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/E/strided_slice_2/stack_1Pack gradient_tape/E/Const_2:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/E/strided_slice_2/stack_2Pack gradient_tape/E/Const_3:output:0*
N*
T0*
_output_shapes
:Ё
gradient_tape/E/strided_slice_2StridedSlicegradient_tape/E/Cast_1:y:0.gradient_tape/E/strided_slice_2/stack:output:00gradient_tape/E/strided_slice_2/stack_1:output:00gradient_tape/E/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_masko
%gradient_tape/E/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:q
'gradient_tape/E/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'gradient_tape/E/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
gradient_tape/E/strided_slice_3StridedSlice(gradient_tape/E/strided_slice_2:output:0.gradient_tape/E/strided_slice_3/stack:output:00gradient_tape/E/strided_slice_3/stack_1:output:00gradient_tape/E/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskt
!gradient_tape/E/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
gradient_tape/E/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
gradient_tape/E/concat_1ConcatV2(gradient_tape/E/strided_slice_1:output:0*gradient_tape/E/concat_1/values_1:output:0(gradient_tape/E/strided_slice_3:output:0&gradient_tape/E/concat_1/axis:output:0*
N*
T0*
_output_shapes
:X
gradient_tape/E/Size_2Const*
_output_shapes
: *
dtype0*
value	B :X
gradient_tape/E/Size_3Const*
_output_shapes
: *
dtype0*
value	B :]
gradient_tape/E/range/startConst*
_output_shapes
: *
dtype0*
value	B : ]
gradient_tape/E/range/limitConst*
_output_shapes
: *
dtype0*
value	B : ]
gradient_tape/E/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Њ
gradient_tape/E/rangeRange$gradient_tape/E/range/start:output:0$gradient_tape/E/range/limit:output:0$gradient_tape/E/range/delta:output:0*
_output_shapes
: _
gradient_tape/E/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/E/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :­
gradient_tape/E/range_1Range&gradient_tape/E/range_1/start:output:0gradient_tape/E/Size_3:output:0&gradient_tape/E/range_1/delta:output:0*
_output_shapes
:W
gradient_tape/E/add/yConst*
_output_shapes
: *
dtype0*
value	B :~
gradient_tape/E/addAddV2gradient_tape/E/Size_3:output:0gradient_tape/E/add/y:output:0*
T0*
_output_shapes
: _
gradient_tape/E/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/E/range_2Rangegradient_tape/E/add:z:0gradient_tape/E/Size_2:output:0&gradient_tape/E/range_2/delta:output:0*
_output_shapes
: Џ
gradient_tape/E/Reshape_2Reshape'gradient_tape/E/ein_A/Einsum_1:output:0!gradient_tape/E/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџx
!gradient_tape/E/concat_2/values_1Packgradient_tape/E/Size_3:output:0*
N*
T0*
_output_shapes
:_
gradient_tape/E/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E/concat_2ConcatV2gradient_tape/E/range:output:0*gradient_tape/E/concat_2/values_1:output:0 gradient_tape/E/range_1:output:0 gradient_tape/E/range_2:output:0&gradient_tape/E/concat_2/axis:output:0*
N*
T0*
_output_shapes
:Ќ
gradient_tape/E/transpose	Transpose"gradient_tape/E/Reshape_2:output:0!gradient_tape/E/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ_
gradient_tape/E/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : г
gradient_tape/E/GatherV2GatherV2gradient_tape/E/Cast_1:y:0!gradient_tape/E/concat_2:output:0&gradient_tape/E/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
gradient_tape/E/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
gradient_tape/E/add_1AddV2E/GatherV2/axis:output:0 gradient_tape/E/add_1/y:output:0*
T0*
_output_shapes
: u
%gradient_tape/E/strided_slice_4/stackPackE/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:x
'gradient_tape/E/strided_slice_4/stack_1Packgradient_tape/E/add_1:z:0*
N*
T0*
_output_shapes
:q
'gradient_tape/E/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѕ
gradient_tape/E/strided_slice_4StridedSlicegradient_tape/E/Cast_1:y:0.gradient_tape/E/strided_slice_4/stack:output:00gradient_tape/E/strided_slice_4/stack_1:output:00gradient_tape/E/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
gradient_tape/E/Size_4Const*
_output_shapes
: *
dtype0*
value	B :b
 gradient_tape/E/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E/ExpandDims_2
ExpandDimsgradient_tape/E/Size_4:output:0)gradient_tape/E/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:
gradient_tape/E/Reshape_3ReshapeE/GatherV2/indices:output:0%gradient_tape/E/ExpandDims_2:output:0*
T0	*
_output_shapes
:є
"gradient_tape/E/UnsortedSegmentSumUnsortedSegmentSumgradient_tape/E/transpose:y:0"gradient_tape/E/Reshape_3:output:0(gradient_tape/E/strided_slice_4:output:0*
Tindices0	*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџY
gradient_tape/E/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/E/add_2AddV2 gradient_tape/E/range_1:output:0 gradient_tape/E/add_2/y:output:0*
T0*
_output_shapes
:k
!gradient_tape/E/concat_3/values_2Const*
_output_shapes
:*
dtype0*
valueB: _
gradient_tape/E/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E/concat_3ConcatV2gradient_tape/E/range:output:0gradient_tape/E/add_2:z:0*gradient_tape/E/concat_3/values_2:output:0 gradient_tape/E/range_2:output:0&gradient_tape/E/concat_3/axis:output:0*
N*
T0*
_output_shapes
:З
gradient_tape/E/transpose_1	Transpose+gradient_tape/E/UnsortedSegmentSum:output:0!gradient_tape/E/concat_3:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџВ
gradient_tape/E2/Shape_2Const*/
_class%
#!loc:@E2/GatherV2_1/ReadVariableOp*
_output_shapes
:*
dtype0	*5
value,B*	"                             Ї
gradient_tape/E2/Cast_2Cast!gradient_tape/E2/Shape_2:output:0*

DstT0*

SrcT0	*/
_class%
#!loc:@E2/GatherV2_1/ReadVariableOp*
_output_shapes
:Y
gradient_tape/E2/Size_5Const*
_output_shapes
: *
dtype0*
value	B :c
!gradient_tape/E2/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E2/ExpandDims_3
ExpandDims gradient_tape/E2/Size_5:output:0*gradient_tape/E2/ExpandDims_3/dim:output:0*
T0*
_output_shapes
:Z
gradient_tape/E2/Const_4Const*
_output_shapes
: *
dtype0*
value	B : Z
gradient_tape/E2/Const_5Const*
_output_shapes
: *
dtype0*
value	B :
&gradient_tape/E2/strided_slice_5/stackPack!gradient_tape/E2/Const_4:output:0*
N*
T0*
_output_shapes
:{
(gradient_tape/E2/strided_slice_5/stack_1PackE2/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:
(gradient_tape/E2/strided_slice_5/stack_2Pack!gradient_tape/E2/Const_5:output:0*
N*
T0*
_output_shapes
:Ј
 gradient_tape/E2/strided_slice_5StridedSlicegradient_tape/E2/Cast_2:y:0/gradient_tape/E2/strided_slice_5/stack:output:01gradient_tape/E2/strided_slice_5/stack_1:output:01gradient_tape/E2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskZ
gradient_tape/E2/Const_6Const*
_output_shapes
: *
dtype0*
value	B : Z
gradient_tape/E2/Const_7Const*
_output_shapes
: *
dtype0*
value	B :y
&gradient_tape/E2/strided_slice_6/stackPackE2/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:
(gradient_tape/E2/strided_slice_6/stack_1Pack!gradient_tape/E2/Const_6:output:0*
N*
T0*
_output_shapes
:
(gradient_tape/E2/strided_slice_6/stack_2Pack!gradient_tape/E2/Const_7:output:0*
N*
T0*
_output_shapes
:І
 gradient_tape/E2/strided_slice_6StridedSlicegradient_tape/E2/Cast_2:y:0/gradient_tape/E2/strided_slice_6/stack:output:01gradient_tape/E2/strided_slice_6/stack_1:output:01gradient_tape/E2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskp
&gradient_tape/E2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(gradient_tape/E2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(gradient_tape/E2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:В
 gradient_tape/E2/strided_slice_7StridedSlice)gradient_tape/E2/strided_slice_6:output:0/gradient_tape/E2/strided_slice_7/stack:output:01gradient_tape/E2/strided_slice_7/stack_1:output:01gradient_tape/E2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masku
"gradient_tape/E2/concat_4/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ`
gradient_tape/E2/concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
gradient_tape/E2/concat_4ConcatV2)gradient_tape/E2/strided_slice_5:output:0+gradient_tape/E2/concat_4/values_1:output:0)gradient_tape/E2/strided_slice_7:output:0'gradient_tape/E2/concat_4/axis:output:0*
N*
T0*
_output_shapes
:Y
gradient_tape/E2/Size_6Const*
_output_shapes
: *
dtype0*
value	B :Y
gradient_tape/E2/Size_7Const*
_output_shapes
: *
dtype0*
value	B :`
gradient_tape/E2/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : `
gradient_tape/E2/range_3/limitConst*
_output_shapes
: *
dtype0*
value	B : `
gradient_tape/E2/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ж
gradient_tape/E2/range_3Range'gradient_tape/E2/range_3/start:output:0'gradient_tape/E2/range_3/limit:output:0'gradient_tape/E2/range_3/delta:output:0*
_output_shapes
: `
gradient_tape/E2/range_4/startConst*
_output_shapes
: *
dtype0*
value	B : `
gradient_tape/E2/range_4/deltaConst*
_output_shapes
: *
dtype0*
value	B :Б
gradient_tape/E2/range_4Range'gradient_tape/E2/range_4/start:output:0 gradient_tape/E2/Size_7:output:0'gradient_tape/E2/range_4/delta:output:0*
_output_shapes
:Z
gradient_tape/E2/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/E2/add_3AddV2 gradient_tape/E2/Size_7:output:0!gradient_tape/E2/add_3/y:output:0*
T0*
_output_shapes
: `
gradient_tape/E2/range_5/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ђ
gradient_tape/E2/range_5Rangegradient_tape/E2/add_3:z:0 gradient_tape/E2/Size_6:output:0'gradient_tape/E2/range_5/delta:output:0*
_output_shapes
: p
&gradient_tape/E2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(gradient_tape/E2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(gradient_tape/E2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ј
 gradient_tape/E2/strided_slice_8StridedSlicegradient_tape/E2/Cast:y:0/gradient_tape/E2/strided_slice_8/stack:output:01gradient_tape/E2/strided_slice_8/stack_1:output:01gradient_tape/E2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskя
%gradient_tape/E2/UnsortedSegmentSum_1UnsortedSegmentSum!gradient_tape/E2/Reshape:output:0#gradient_tape/E2/Reshape_1:output:0)gradient_tape/E2/strided_slice_8:output:0*
Tindices0*
T0*&
_output_shapes
:Њ
gradient_tape/E2/Reshape_4Reshape.gradient_tape/E2/UnsortedSegmentSum_1:output:0"gradient_tape/E2/concat_4:output:0*
T0*&
_output_shapes
:z
"gradient_tape/E2/concat_5/values_1Pack gradient_tape/E2/Size_7:output:0*
N*
T0*
_output_shapes
:`
gradient_tape/E2/concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E2/concat_5ConcatV2!gradient_tape/E2/range_3:output:0+gradient_tape/E2/concat_5/values_1:output:0!gradient_tape/E2/range_4:output:0!gradient_tape/E2/range_5:output:0'gradient_tape/E2/concat_5/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
gradient_tape/E2/transpose_2	Transpose#gradient_tape/E2/Reshape_4:output:0"gradient_tape/E2/concat_5:output:0*
T0*&
_output_shapes
:b
 gradient_tape/E2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : л
gradient_tape/E2/GatherV2_1GatherV2gradient_tape/E2/Cast_2:y:0"gradient_tape/E2/concat_5:output:0)gradient_tape/E2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Z
gradient_tape/E2/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/E2/add_4AddV2E2/GatherV2_1/axis:output:0!gradient_tape/E2/add_4/y:output:0*
T0*
_output_shapes
: y
&gradient_tape/E2/strided_slice_9/stackPackE2/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:z
(gradient_tape/E2/strided_slice_9/stack_1Packgradient_tape/E2/add_4:z:0*
N*
T0*
_output_shapes
:r
(gradient_tape/E2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
 gradient_tape/E2/strided_slice_9StridedSlicegradient_tape/E2/Cast_2:y:0/gradient_tape/E2/strided_slice_9/stack:output:01gradient_tape/E2/strided_slice_9/stack_1:output:01gradient_tape/E2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
gradient_tape/E2/Size_8Const*
_output_shapes
: *
dtype0*
value	B :c
!gradient_tape/E2/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E2/ExpandDims_4
ExpandDims gradient_tape/E2/Size_8:output:0*gradient_tape/E2/ExpandDims_4/dim:output:0*
T0*
_output_shapes
:
gradient_tape/E2/Reshape_5Reshapee2_gatherv2_1_indices&gradient_tape/E2/ExpandDims_4:output:0*
T0*
_output_shapes
:ю
%gradient_tape/E2/UnsortedSegmentSum_2UnsortedSegmentSum gradient_tape/E2/transpose_2:y:0#gradient_tape/E2/Reshape_5:output:0)gradient_tape/E2/strided_slice_9:output:0*
Tindices0*
T0*&
_output_shapes
:Z
gradient_tape/E2/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/E2/add_5AddV2!gradient_tape/E2/range_4:output:0!gradient_tape/E2/add_5/y:output:0*
T0*
_output_shapes
:l
"gradient_tape/E2/concat_6/values_2Const*
_output_shapes
:*
dtype0*
valueB: `
gradient_tape/E2/concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E2/concat_6ConcatV2!gradient_tape/E2/range_3:output:0gradient_tape/E2/add_5:z:0+gradient_tape/E2/concat_6/values_2:output:0!gradient_tape/E2/range_5:output:0'gradient_tape/E2/concat_6/axis:output:0*
N*
T0*
_output_shapes
:Ў
gradient_tape/E2/transpose_3	Transpose.gradient_tape/E2/UnsortedSegmentSum_2:output:0"gradient_tape/E2/concat_6:output:0*
T0*&
_output_shapes
:\
gradient_tape/add_3/ShapeConst*
_output_shapes
: *
dtype0*
valueB q
gradient_tape/add_3/Shape_1Shapestrided_slice_2:output:0*
T0*
_output_shapes
::эЯs
#gradient_tape/strided_slice_3/ShapeShaperho2/transpose:y:0*
T0*
_output_shapes
::эЯ
4gradient_tape/strided_slice_3/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*!
valueB"            
2gradient_tape/strided_slice_3/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*!
valueB"           
6gradient_tape/strided_slice_3/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*!
valueB"         в
.gradient_tape/strided_slice_3/StridedSliceGradStridedSliceGrad,gradient_tape/strided_slice_3/Shape:output:0=gradient_tape/strided_slice_3/StridedSliceGrad/begin:output:0;gradient_tape/strided_slice_3/StridedSliceGrad/end:output:0?gradient_tape/strided_slice_3/StridedSliceGrad/strides:output:0&gradient_tape/add_4/Reshape_1:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskА
gradient_tape/E/Shape_2Const*.
_class$
" loc:@E/GatherV2_1/ReadVariableOp*
_output_shapes
:*
dtype0	*5
value,B*	"                             Є
gradient_tape/E/Cast_2Cast gradient_tape/E/Shape_2:output:0*

DstT0*

SrcT0	*.
_class$
" loc:@E/GatherV2_1/ReadVariableOp*
_output_shapes
:X
gradient_tape/E/Size_5Const*
_output_shapes
: *
dtype0*
value	B :b
 gradient_tape/E/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E/ExpandDims_3
ExpandDimsgradient_tape/E/Size_5:output:0)gradient_tape/E/ExpandDims_3/dim:output:0*
T0*
_output_shapes
:Y
gradient_tape/E/Const_4Const*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/E/Const_5Const*
_output_shapes
: *
dtype0*
value	B :}
%gradient_tape/E/strided_slice_5/stackPack gradient_tape/E/Const_4:output:0*
N*
T0*
_output_shapes
:y
'gradient_tape/E/strided_slice_5/stack_1PackE/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/E/strided_slice_5/stack_2Pack gradient_tape/E/Const_5:output:0*
N*
T0*
_output_shapes
:Ѓ
gradient_tape/E/strided_slice_5StridedSlicegradient_tape/E/Cast_2:y:0.gradient_tape/E/strided_slice_5/stack:output:00gradient_tape/E/strided_slice_5/stack_1:output:00gradient_tape/E/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskY
gradient_tape/E/Const_6Const*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/E/Const_7Const*
_output_shapes
: *
dtype0*
value	B :w
%gradient_tape/E/strided_slice_6/stackPackE/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/E/strided_slice_6/stack_1Pack gradient_tape/E/Const_6:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/E/strided_slice_6/stack_2Pack gradient_tape/E/Const_7:output:0*
N*
T0*
_output_shapes
:Ё
gradient_tape/E/strided_slice_6StridedSlicegradient_tape/E/Cast_2:y:0.gradient_tape/E/strided_slice_6/stack:output:00gradient_tape/E/strided_slice_6/stack_1:output:00gradient_tape/E/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_masko
%gradient_tape/E/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:q
'gradient_tape/E/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'gradient_tape/E/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
gradient_tape/E/strided_slice_7StridedSlice(gradient_tape/E/strided_slice_6:output:0.gradient_tape/E/strided_slice_7/stack:output:00gradient_tape/E/strided_slice_7/stack_1:output:00gradient_tape/E/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskt
!gradient_tape/E/concat_4/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
gradient_tape/E/concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
gradient_tape/E/concat_4ConcatV2(gradient_tape/E/strided_slice_5:output:0*gradient_tape/E/concat_4/values_1:output:0(gradient_tape/E/strided_slice_7:output:0&gradient_tape/E/concat_4/axis:output:0*
N*
T0*
_output_shapes
:X
gradient_tape/E/Size_6Const*
_output_shapes
: *
dtype0*
value	B :X
gradient_tape/E/Size_7Const*
_output_shapes
: *
dtype0*
value	B :_
gradient_tape/E/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/E/range_3/limitConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/E/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :В
gradient_tape/E/range_3Range&gradient_tape/E/range_3/start:output:0&gradient_tape/E/range_3/limit:output:0&gradient_tape/E/range_3/delta:output:0*
_output_shapes
: _
gradient_tape/E/range_4/startConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/E/range_4/deltaConst*
_output_shapes
: *
dtype0*
value	B :­
gradient_tape/E/range_4Range&gradient_tape/E/range_4/start:output:0gradient_tape/E/Size_7:output:0&gradient_tape/E/range_4/delta:output:0*
_output_shapes
:Y
gradient_tape/E/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/E/add_3AddV2gradient_tape/E/Size_7:output:0 gradient_tape/E/add_3/y:output:0*
T0*
_output_shapes
: _
gradient_tape/E/range_5/deltaConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/E/range_5Rangegradient_tape/E/add_3:z:0gradient_tape/E/Size_6:output:0&gradient_tape/E/range_5/delta:output:0*
_output_shapes
: o
%gradient_tape/E/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'gradient_tape/E/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'gradient_tape/E/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
gradient_tape/E/strided_slice_8StridedSlicegradient_tape/E/Cast:y:0.gradient_tape/E/strided_slice_8/stack:output:00gradient_tape/E/strided_slice_8/stack_1:output:00gradient_tape/E/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskы
$gradient_tape/E/UnsortedSegmentSum_1UnsortedSegmentSum gradient_tape/E/Reshape:output:0"gradient_tape/E/Reshape_1:output:0(gradient_tape/E/strided_slice_8:output:0*
Tindices0*
T0*&
_output_shapes
:Ї
gradient_tape/E/Reshape_4Reshape-gradient_tape/E/UnsortedSegmentSum_1:output:0!gradient_tape/E/concat_4:output:0*
T0*&
_output_shapes
:x
!gradient_tape/E/concat_5/values_1Packgradient_tape/E/Size_7:output:0*
N*
T0*
_output_shapes
:_
gradient_tape/E/concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E/concat_5ConcatV2 gradient_tape/E/range_3:output:0*gradient_tape/E/concat_5/values_1:output:0 gradient_tape/E/range_4:output:0 gradient_tape/E/range_5:output:0&gradient_tape/E/concat_5/axis:output:0*
N*
T0*
_output_shapes
: 
gradient_tape/E/transpose_2	Transpose"gradient_tape/E/Reshape_4:output:0!gradient_tape/E/concat_5:output:0*
T0*&
_output_shapes
:a
gradient_tape/E/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
gradient_tape/E/GatherV2_1GatherV2gradient_tape/E/Cast_2:y:0!gradient_tape/E/concat_5:output:0(gradient_tape/E/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
gradient_tape/E/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :}
gradient_tape/E/add_4AddV2E/GatherV2_1/axis:output:0 gradient_tape/E/add_4/y:output:0*
T0*
_output_shapes
: w
%gradient_tape/E/strided_slice_9/stackPackE/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:x
'gradient_tape/E/strided_slice_9/stack_1Packgradient_tape/E/add_4:z:0*
N*
T0*
_output_shapes
:q
'gradient_tape/E/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѕ
gradient_tape/E/strided_slice_9StridedSlicegradient_tape/E/Cast_2:y:0.gradient_tape/E/strided_slice_9/stack:output:00gradient_tape/E/strided_slice_9/stack_1:output:00gradient_tape/E/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
gradient_tape/E/Size_8Const*
_output_shapes
: *
dtype0*
value	B :b
 gradient_tape/E/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E/ExpandDims_4
ExpandDimsgradient_tape/E/Size_8:output:0)gradient_tape/E/ExpandDims_4/dim:output:0*
T0*
_output_shapes
:
gradient_tape/E/Reshape_5Reshapee_gatherv2_1_indices%gradient_tape/E/ExpandDims_4:output:0*
T0*
_output_shapes
:ъ
$gradient_tape/E/UnsortedSegmentSum_2UnsortedSegmentSumgradient_tape/E/transpose_2:y:0"gradient_tape/E/Reshape_5:output:0(gradient_tape/E/strided_slice_9:output:0*
Tindices0*
T0*&
_output_shapes
:Y
gradient_tape/E/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/E/add_5AddV2 gradient_tape/E/range_4:output:0 gradient_tape/E/add_5/y:output:0*
T0*
_output_shapes
:k
!gradient_tape/E/concat_6/values_2Const*
_output_shapes
:*
dtype0*
valueB: _
gradient_tape/E/concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/E/concat_6ConcatV2 gradient_tape/E/range_3:output:0gradient_tape/E/add_5:z:0*gradient_tape/E/concat_6/values_2:output:0 gradient_tape/E/range_5:output:0&gradient_tape/E/concat_6/axis:output:0*
N*
T0*
_output_shapes
:Ћ
gradient_tape/E/transpose_3	Transpose-gradient_tape/E/UnsortedSegmentSum_2:output:0!gradient_tape/E/concat_6:output:0*
T0*&
_output_shapes
:r
#gradient_tape/strided_slice_2/ShapeShaperho/transpose:y:0*
T0*
_output_shapes
::эЯ
4gradient_tape/strided_slice_2/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*!
valueB"            
2gradient_tape/strided_slice_2/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*!
valueB"           
6gradient_tape/strided_slice_2/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*!
valueB"         а
.gradient_tape/strided_slice_2/StridedSliceGradStridedSliceGrad,gradient_tape/strided_slice_2/Shape:output:0=gradient_tape/strided_slice_2/StridedSliceGrad/begin:output:0;gradient_tape/strided_slice_2/StridedSliceGrad/end:output:0?gradient_tape/strided_slice_2/StridedSliceGrad/strides:output:0$gradient_tape/add_4/Reshape:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask}
.gradient_tape/rho2/transpose/InvertPermutationInvertPermutationrho2/transpose/perm:output:0*
_output_shapes
:ж
&gradient_tape/rho2/transpose/transpose	Transpose7gradient_tape/strided_slice_3/StridedSliceGrad:output:02gradient_tape/rho2/transpose/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџ{
-gradient_tape/rho/transpose/InvertPermutationInvertPermutationrho/transpose/perm:output:0*
_output_shapes
:д
%gradient_tape/rho/transpose/transpose	Transpose7gradient_tape/strided_slice_2/StridedSliceGrad:output:01gradient_tape/rho/transpose/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџО
gradient_tape/rho2/GatherNdGatherNd*gradient_tape/rho2/transpose/transpose:y:0rho2_tensorscatteradd_indices*
Tindices0*
Tparams0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/rho2/IdentityIdentity*gradient_tape/rho2/transpose/transpose:y:0*
T0*+
_output_shapes
:џџџџџџџџџЛ
gradient_tape/rho/GatherNdGatherNd)gradient_tape/rho/transpose/transpose:y:0rho_tensorscatteradd_indices*
Tindices0*
Tparams0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/rho/IdentityIdentity)gradient_tape/rho/transpose/transpose:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/rho2/mul/MulMul$gradient_tape/rho2/GatherNd:output:0
rho2_mul_y*
T0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/rho/mul/MulMul#gradient_tape/rho/GatherNd:output:0	rho_mul_y*
T0*+
_output_shapes
:џџџџџџџџџt
gradient_tape/rho2/ein_B/ShapeShaperho2/GatherV2_2:output:0*
T0*
_output_shapes
::эЯt
 gradient_tape/rho2/ein_B/Shape_1Shaperho2/GatherV2:output:0*
T0*
_output_shapes
::эЯС
gradient_tape/rho2/ein_B/EinsumEinsumgradient_tape/rho2/mul/Mul:z:0rho2/GatherV2:output:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ*
equationwak,anw->aknwС
!gradient_tape/rho2/ein_B/Einsum_1Einsumgradient_tape/rho2/mul/Mul:z:0rho2/GatherV2_2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationwak,aknw->anwr
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
:џџџџџџџџџ*
equationwak,anw->aknwО
 gradient_tape/rho/ein_A/Einsum_1Einsumgradient_tape/rho/mul/Mul:z:0rho/GatherV2_2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationwak,aknw->anwЅ
gradient_tape/rho2/ShapeConst*"
_class
loc:@rho2/GatherV2_1*
_output_shapes
:*
dtype0	*5
value,B*	"                             
gradient_tape/rho2/CastCast!gradient_tape/rho2/Shape:output:0*

DstT0*

SrcT0	*"
_class
loc:@rho2/GatherV2_1*
_output_shapes
:M
gradient_tape/rho2/SizeSizeatomic_mu_i*
T0*
_output_shapes
: c
!gradient_tape/rho2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho2/ExpandDims
ExpandDims gradient_tape/rho2/Size:output:0*gradient_tape/rho2/ExpandDims/dim:output:0*
T0*
_output_shapes
:p
&gradient_tape/rho2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(gradient_tape/rho2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(gradient_tape/rho2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:І
 gradient_tape/rho2/strided_sliceStridedSlicegradient_tape/rho2/Cast:y:0/gradient_tape/rho2/strided_slice/stack:output:01gradient_tape/rho2/strided_slice/stack_1:output:01gradient_tape/rho2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask`
gradient_tape/rho2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
gradient_tape/rho2/concatConcatV2&gradient_tape/rho2/ExpandDims:output:0)gradient_tape/rho2/strided_slice:output:0'gradient_tape/rho2/concat/axis:output:0*
N*
T0*
_output_shapes
:­
gradient_tape/rho2/ReshapeReshape(gradient_tape/rho2/ein_B/Einsum:output:0"gradient_tape/rho2/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
gradient_tape/rho2/Reshape_1Reshapeatomic_mu_i&gradient_tape/rho2/ExpandDims:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/rho2/Shape_1ShapeB/transpose:y:0*
T0*
_class
loc:@B/transpose*
_output_shapes
:*
out_type0	:эа
gradient_tape/rho2/Cast_1Cast#gradient_tape/rho2/Shape_1:output:0*

DstT0*

SrcT0	*
_class
loc:@B/transpose*
_output_shapes
:[
gradient_tape/rho2/Size_1Const*
_output_shapes
: *
dtype0*
value	B :e
#gradient_tape/rho2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Є
gradient_tape/rho2/ExpandDims_1
ExpandDims"gradient_tape/rho2/Size_1:output:0,gradient_tape/rho2/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:Z
gradient_tape/rho2/ConstConst*
_output_shapes
: *
dtype0*
value	B : \
gradient_tape/rho2/Const_1Const*
_output_shapes
: *
dtype0*
value	B :
(gradient_tape/rho2/strided_slice_1/stackPack!gradient_tape/rho2/Const:output:0*
N*
T0*
_output_shapes
:}
*gradient_tape/rho2/strided_slice_1/stack_1Packrho2/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:
*gradient_tape/rho2/strided_slice_1/stack_2Pack#gradient_tape/rho2/Const_1:output:0*
N*
T0*
_output_shapes
:В
"gradient_tape/rho2/strided_slice_1StridedSlicegradient_tape/rho2/Cast_1:y:01gradient_tape/rho2/strided_slice_1/stack:output:03gradient_tape/rho2/strided_slice_1/stack_1:output:03gradient_tape/rho2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask\
gradient_tape/rho2/Const_2Const*
_output_shapes
: *
dtype0*
value	B : \
gradient_tape/rho2/Const_3Const*
_output_shapes
: *
dtype0*
value	B :{
(gradient_tape/rho2/strided_slice_2/stackPackrho2/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:
*gradient_tape/rho2/strided_slice_2/stack_1Pack#gradient_tape/rho2/Const_2:output:0*
N*
T0*
_output_shapes
:
*gradient_tape/rho2/strided_slice_2/stack_2Pack#gradient_tape/rho2/Const_3:output:0*
N*
T0*
_output_shapes
:А
"gradient_tape/rho2/strided_slice_2StridedSlicegradient_tape/rho2/Cast_1:y:01gradient_tape/rho2/strided_slice_2/stack:output:03gradient_tape/rho2/strided_slice_2/stack_1:output:03gradient_tape/rho2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(gradient_tape/rho2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*gradient_tape/rho2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*gradient_tape/rho2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:М
"gradient_tape/rho2/strided_slice_3StridedSlice+gradient_tape/rho2/strided_slice_2:output:01gradient_tape/rho2/strided_slice_3/stack:output:03gradient_tape/rho2/strided_slice_3/stack_1:output:03gradient_tape/rho2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskw
$gradient_tape/rho2/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
 gradient_tape/rho2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho2/concat_1ConcatV2+gradient_tape/rho2/strided_slice_1:output:0-gradient_tape/rho2/concat_1/values_1:output:0+gradient_tape/rho2/strided_slice_3:output:0)gradient_tape/rho2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:[
gradient_tape/rho2/Size_2Const*
_output_shapes
: *
dtype0*
value	B :[
gradient_tape/rho2/Size_3Const*
_output_shapes
: *
dtype0*
value	B :`
gradient_tape/rho2/range/startConst*
_output_shapes
: *
dtype0*
value	B : `
gradient_tape/rho2/range/limitConst*
_output_shapes
: *
dtype0*
value	B : `
gradient_tape/rho2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ж
gradient_tape/rho2/rangeRange'gradient_tape/rho2/range/start:output:0'gradient_tape/rho2/range/limit:output:0'gradient_tape/rho2/range/delta:output:0*
_output_shapes
: b
 gradient_tape/rho2/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : b
 gradient_tape/rho2/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :Й
gradient_tape/rho2/range_1Range)gradient_tape/rho2/range_1/start:output:0"gradient_tape/rho2/Size_3:output:0)gradient_tape/rho2/range_1/delta:output:0*
_output_shapes
:Z
gradient_tape/rho2/add/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/rho2/addAddV2"gradient_tape/rho2/Size_3:output:0!gradient_tape/rho2/add/y:output:0*
T0*
_output_shapes
: b
 gradient_tape/rho2/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ј
gradient_tape/rho2/range_2Rangegradient_tape/rho2/add:z:0"gradient_tape/rho2/Size_2:output:0)gradient_tape/rho2/range_2/delta:output:0*
_output_shapes
: И
gradient_tape/rho2/Reshape_2Reshape*gradient_tape/rho2/ein_B/Einsum_1:output:0$gradient_tape/rho2/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ~
$gradient_tape/rho2/concat_2/values_1Pack"gradient_tape/rho2/Size_3:output:0*
N*
T0*
_output_shapes
:b
 gradient_tape/rho2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho2/concat_2ConcatV2!gradient_tape/rho2/range:output:0-gradient_tape/rho2/concat_2/values_1:output:0#gradient_tape/rho2/range_1:output:0#gradient_tape/rho2/range_2:output:0)gradient_tape/rho2/concat_2/axis:output:0*
N*
T0*
_output_shapes
:Е
gradient_tape/rho2/transpose	Transpose%gradient_tape/rho2/Reshape_2:output:0$gradient_tape/rho2/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџb
 gradient_tape/rho2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : п
gradient_tape/rho2/GatherV2GatherV2gradient_tape/rho2/Cast_1:y:0$gradient_tape/rho2/concat_2:output:0)gradient_tape/rho2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:\
gradient_tape/rho2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/rho2/add_1AddV2rho2/GatherV2/axis:output:0#gradient_tape/rho2/add_1/y:output:0*
T0*
_output_shapes
: {
(gradient_tape/rho2/strided_slice_4/stackPackrho2/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:~
*gradient_tape/rho2/strided_slice_4/stack_1Packgradient_tape/rho2/add_1:z:0*
N*
T0*
_output_shapes
:t
*gradient_tape/rho2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Д
"gradient_tape/rho2/strided_slice_4StridedSlicegradient_tape/rho2/Cast_1:y:01gradient_tape/rho2/strided_slice_4/stack:output:03gradient_tape/rho2/strided_slice_4/stack_1:output:03gradient_tape/rho2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
gradient_tape/rho2/Size_4Const*
_output_shapes
: *
dtype0*
value	B :e
#gradient_tape/rho2/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : Є
gradient_tape/rho2/ExpandDims_2
ExpandDims"gradient_tape/rho2/Size_4:output:0,gradient_tape/rho2/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:
gradient_tape/rho2/Reshape_3Reshaperho2/GatherV2/indices:output:0(gradient_tape/rho2/ExpandDims_2:output:0*
T0	*
_output_shapes
:
%gradient_tape/rho2/UnsortedSegmentSumUnsortedSegmentSum gradient_tape/rho2/transpose:y:0%gradient_tape/rho2/Reshape_3:output:0+gradient_tape/rho2/strided_slice_4:output:0*
Tindices0	*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ\
gradient_tape/rho2/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/rho2/add_2AddV2#gradient_tape/rho2/range_1:output:0#gradient_tape/rho2/add_2/y:output:0*
T0*
_output_shapes
:n
$gradient_tape/rho2/concat_3/values_2Const*
_output_shapes
:*
dtype0*
valueB: b
 gradient_tape/rho2/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho2/concat_3ConcatV2!gradient_tape/rho2/range:output:0gradient_tape/rho2/add_2:z:0-gradient_tape/rho2/concat_3/values_2:output:0#gradient_tape/rho2/range_2:output:0)gradient_tape/rho2/concat_3/axis:output:0*
N*
T0*
_output_shapes
:Р
gradient_tape/rho2/transpose_1	Transpose.gradient_tape/rho2/UnsortedSegmentSum:output:0$gradient_tape/rho2/concat_3:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЃ
gradient_tape/rho/ShapeConst*!
_class
loc:@rho/GatherV2_1*
_output_shapes
:*
dtype0	*5
value,B*	"                             
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
:џџџџџџџџџ
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
 :џџџџџџџџџџџџџџџџџџЖ
gradient_tape/rho2/Shape_2Const*1
_class'
%#loc:@rho2/GatherV2_1/ReadVariableOp*
_output_shapes
:*
dtype0	*5
value,B*	"                             ­
gradient_tape/rho2/Cast_2Cast#gradient_tape/rho2/Shape_2:output:0*

DstT0*

SrcT0	*1
_class'
%#loc:@rho2/GatherV2_1/ReadVariableOp*
_output_shapes
:[
gradient_tape/rho2/Size_5Const*
_output_shapes
: *
dtype0*
value	B :e
#gradient_tape/rho2/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : Є
gradient_tape/rho2/ExpandDims_3
ExpandDims"gradient_tape/rho2/Size_5:output:0,gradient_tape/rho2/ExpandDims_3/dim:output:0*
T0*
_output_shapes
:\
gradient_tape/rho2/Const_4Const*
_output_shapes
: *
dtype0*
value	B : \
gradient_tape/rho2/Const_5Const*
_output_shapes
: *
dtype0*
value	B :
(gradient_tape/rho2/strided_slice_5/stackPack#gradient_tape/rho2/Const_4:output:0*
N*
T0*
_output_shapes
:
*gradient_tape/rho2/strided_slice_5/stack_1Packrho2/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:
*gradient_tape/rho2/strided_slice_5/stack_2Pack#gradient_tape/rho2/Const_5:output:0*
N*
T0*
_output_shapes
:В
"gradient_tape/rho2/strided_slice_5StridedSlicegradient_tape/rho2/Cast_2:y:01gradient_tape/rho2/strided_slice_5/stack:output:03gradient_tape/rho2/strided_slice_5/stack_1:output:03gradient_tape/rho2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask\
gradient_tape/rho2/Const_6Const*
_output_shapes
: *
dtype0*
value	B : \
gradient_tape/rho2/Const_7Const*
_output_shapes
: *
dtype0*
value	B :}
(gradient_tape/rho2/strided_slice_6/stackPackrho2/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:
*gradient_tape/rho2/strided_slice_6/stack_1Pack#gradient_tape/rho2/Const_6:output:0*
N*
T0*
_output_shapes
:
*gradient_tape/rho2/strided_slice_6/stack_2Pack#gradient_tape/rho2/Const_7:output:0*
N*
T0*
_output_shapes
:А
"gradient_tape/rho2/strided_slice_6StridedSlicegradient_tape/rho2/Cast_2:y:01gradient_tape/rho2/strided_slice_6/stack:output:03gradient_tape/rho2/strided_slice_6/stack_1:output:03gradient_tape/rho2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
(gradient_tape/rho2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*gradient_tape/rho2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*gradient_tape/rho2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:М
"gradient_tape/rho2/strided_slice_7StridedSlice+gradient_tape/rho2/strided_slice_6:output:01gradient_tape/rho2/strided_slice_7/stack:output:03gradient_tape/rho2/strided_slice_7/stack_1:output:03gradient_tape/rho2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskw
$gradient_tape/rho2/concat_4/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
 gradient_tape/rho2/concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho2/concat_4ConcatV2+gradient_tape/rho2/strided_slice_5:output:0-gradient_tape/rho2/concat_4/values_1:output:0+gradient_tape/rho2/strided_slice_7:output:0)gradient_tape/rho2/concat_4/axis:output:0*
N*
T0*
_output_shapes
:[
gradient_tape/rho2/Size_6Const*
_output_shapes
: *
dtype0*
value	B :[
gradient_tape/rho2/Size_7Const*
_output_shapes
: *
dtype0*
value	B :b
 gradient_tape/rho2/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : b
 gradient_tape/rho2/range_3/limitConst*
_output_shapes
: *
dtype0*
value	B : b
 gradient_tape/rho2/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :О
gradient_tape/rho2/range_3Range)gradient_tape/rho2/range_3/start:output:0)gradient_tape/rho2/range_3/limit:output:0)gradient_tape/rho2/range_3/delta:output:0*
_output_shapes
: b
 gradient_tape/rho2/range_4/startConst*
_output_shapes
: *
dtype0*
value	B : b
 gradient_tape/rho2/range_4/deltaConst*
_output_shapes
: *
dtype0*
value	B :Й
gradient_tape/rho2/range_4Range)gradient_tape/rho2/range_4/start:output:0"gradient_tape/rho2/Size_7:output:0)gradient_tape/rho2/range_4/delta:output:0*
_output_shapes
:\
gradient_tape/rho2/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/rho2/add_3AddV2"gradient_tape/rho2/Size_7:output:0#gradient_tape/rho2/add_3/y:output:0*
T0*
_output_shapes
: b
 gradient_tape/rho2/range_5/deltaConst*
_output_shapes
: *
dtype0*
value	B :Њ
gradient_tape/rho2/range_5Rangegradient_tape/rho2/add_3:z:0"gradient_tape/rho2/Size_6:output:0)gradient_tape/rho2/range_5/delta:output:0*
_output_shapes
: r
(gradient_tape/rho2/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*gradient_tape/rho2/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*gradient_tape/rho2/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:В
"gradient_tape/rho2/strided_slice_8StridedSlicegradient_tape/rho2/Cast:y:01gradient_tape/rho2/strided_slice_8/stack:output:03gradient_tape/rho2/strided_slice_8/stack_1:output:03gradient_tape/rho2/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskї
'gradient_tape/rho2/UnsortedSegmentSum_1UnsortedSegmentSum#gradient_tape/rho2/Reshape:output:0%gradient_tape/rho2/Reshape_1:output:0+gradient_tape/rho2/strided_slice_8:output:0*
Tindices0*
T0*&
_output_shapes
:А
gradient_tape/rho2/Reshape_4Reshape0gradient_tape/rho2/UnsortedSegmentSum_1:output:0$gradient_tape/rho2/concat_4:output:0*
T0*&
_output_shapes
:~
$gradient_tape/rho2/concat_5/values_1Pack"gradient_tape/rho2/Size_7:output:0*
N*
T0*
_output_shapes
:b
 gradient_tape/rho2/concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho2/concat_5ConcatV2#gradient_tape/rho2/range_3:output:0-gradient_tape/rho2/concat_5/values_1:output:0#gradient_tape/rho2/range_4:output:0#gradient_tape/rho2/range_5:output:0)gradient_tape/rho2/concat_5/axis:output:0*
N*
T0*
_output_shapes
:Љ
gradient_tape/rho2/transpose_2	Transpose%gradient_tape/rho2/Reshape_4:output:0$gradient_tape/rho2/concat_5:output:0*
T0*&
_output_shapes
:d
"gradient_tape/rho2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : у
gradient_tape/rho2/GatherV2_1GatherV2gradient_tape/rho2/Cast_2:y:0$gradient_tape/rho2/concat_5:output:0+gradient_tape/rho2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:\
gradient_tape/rho2/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/rho2/add_4AddV2rho2/GatherV2_1/axis:output:0#gradient_tape/rho2/add_4/y:output:0*
T0*
_output_shapes
: }
(gradient_tape/rho2/strided_slice_9/stackPackrho2/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:~
*gradient_tape/rho2/strided_slice_9/stack_1Packgradient_tape/rho2/add_4:z:0*
N*
T0*
_output_shapes
:t
*gradient_tape/rho2/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Д
"gradient_tape/rho2/strided_slice_9StridedSlicegradient_tape/rho2/Cast_2:y:01gradient_tape/rho2/strided_slice_9/stack:output:03gradient_tape/rho2/strided_slice_9/stack_1:output:03gradient_tape/rho2/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
gradient_tape/rho2/Size_8Const*
_output_shapes
: *
dtype0*
value	B :e
#gradient_tape/rho2/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B : Є
gradient_tape/rho2/ExpandDims_4
ExpandDims"gradient_tape/rho2/Size_8:output:0,gradient_tape/rho2/ExpandDims_4/dim:output:0*
T0*
_output_shapes
:
gradient_tape/rho2/Reshape_5Reshaperho2_gatherv2_1_indices(gradient_tape/rho2/ExpandDims_4:output:0*
T0*
_output_shapes
:і
'gradient_tape/rho2/UnsortedSegmentSum_2UnsortedSegmentSum"gradient_tape/rho2/transpose_2:y:0%gradient_tape/rho2/Reshape_5:output:0+gradient_tape/rho2/strided_slice_9:output:0*
Tindices0*
T0*&
_output_shapes
:\
gradient_tape/rho2/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/rho2/add_5AddV2#gradient_tape/rho2/range_4:output:0#gradient_tape/rho2/add_5/y:output:0*
T0*
_output_shapes
:n
$gradient_tape/rho2/concat_6/values_2Const*
_output_shapes
:*
dtype0*
valueB: b
 gradient_tape/rho2/concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/rho2/concat_6ConcatV2#gradient_tape/rho2/range_3:output:0gradient_tape/rho2/add_5:z:0-gradient_tape/rho2/concat_6/values_2:output:0#gradient_tape/rho2/range_5:output:0)gradient_tape/rho2/concat_6/axis:output:0*
N*
T0*
_output_shapes
:Д
gradient_tape/rho2/transpose_3	Transpose0gradient_tape/rho2/UnsortedSegmentSum_2:output:0$gradient_tape/rho2/concat_6:output:0*
T0*&
_output_shapes
:
AddNAddN gradient_tape/E2/transpose_1:y:0"gradient_tape/rho2/transpose_1:y:0*
N*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџw
+gradient_tape/B/transpose/InvertPermutationInvertPermutationB/transpose/perm:output:0*
_output_shapes
:Ќ
#gradient_tape/B/transpose/transpose	Transpose
AddN:sum:0/gradient_tape/B/transpose/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџД
gradient_tape/rho/Shape_2Const*0
_class&
$"loc:@rho/GatherV2_1/ReadVariableOp*
_output_shapes
:*
dtype0	*5
value,B*	"                             Њ
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
:­
gradient_tape/rho/Reshape_4Reshape/gradient_tape/rho/UnsortedSegmentSum_1:output:0#gradient_tape/rho/concat_4:output:0*
T0*&
_output_shapes
:|
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
:c
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
:[
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
:Е
gradient_tape/B/GatherNdGatherNd'gradient_tape/B/transpose/transpose:y:0b_tensorscatteradd_indices*
Tindices0*
Tparams0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/B/IdentityIdentity'gradient_tape/B/transpose/transpose:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
gradient_tape/B/mul/MulMul!gradient_tape/B/GatherNd:output:0b_mul_y*
T0*+
_output_shapes
:џџџџџџџџџq
gradient_tape/B/ein_YI/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         o
gradient_tape/B/ein_YI/Shape_1ShapeB/GatherV2:output:0*
T0*
_output_shapes
::эЯЋ
gradient_tape/B/ein_YI/EinsumEinsumgradient_tape/B/mul/Mul:z:0B/GatherV2:output:0*
N*
T0*"
_output_shapes
:*
equationwak,anw->knwИ
gradient_tape/B/ein_YI/Einsum_1Einsumgradient_tape/B/mul/Mul:z:0B/GatherV2_1:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationwak,knw->anwІ
gradient_tape/B/ShapeConst*.
_class$
" loc:@B/GatherV2_1/ReadVariableOp*
_output_shapes
:*
dtype0	*-
value$B"	"                      
gradient_tape/B/CastCastgradient_tape/B/Shape:output:0*

DstT0*

SrcT0	*.
_class$
" loc:@B/GatherV2_1/ReadVariableOp*
_output_shapes
:V
gradient_tape/B/SizeConst*
_output_shapes
: *
dtype0*
value	B :`
gradient_tape/B/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/B/ExpandDims
ExpandDimsgradient_tape/B/Size:output:0'gradient_tape/B/ExpandDims/dim:output:0*
T0*
_output_shapes
:W
gradient_tape/B/ConstConst*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/B/Const_1Const*
_output_shapes
: *
dtype0*
value	B :y
#gradient_tape/B/strided_slice/stackPackgradient_tape/B/Const:output:0*
N*
T0*
_output_shapes
:w
%gradient_tape/B/strided_slice/stack_1PackB/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:}
%gradient_tape/B/strided_slice/stack_2Pack gradient_tape/B/Const_1:output:0*
N*
T0*
_output_shapes
:
gradient_tape/B/strided_sliceStridedSlicegradient_tape/B/Cast:y:0,gradient_tape/B/strided_slice/stack:output:0.gradient_tape/B/strided_slice/stack_1:output:0.gradient_tape/B/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskY
gradient_tape/B/Const_2Const*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/B/Const_3Const*
_output_shapes
: *
dtype0*
value	B :w
%gradient_tape/B/strided_slice_1/stackPackB/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/B/strided_slice_1/stack_1Pack gradient_tape/B/Const_2:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/B/strided_slice_1/stack_2Pack gradient_tape/B/Const_3:output:0*
N*
T0*
_output_shapes
:
gradient_tape/B/strided_slice_1StridedSlicegradient_tape/B/Cast:y:0.gradient_tape/B/strided_slice_1/stack:output:00gradient_tape/B/strided_slice_1/stack_1:output:00gradient_tape/B/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_masko
%gradient_tape/B/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:q
'gradient_tape/B/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'gradient_tape/B/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
gradient_tape/B/strided_slice_2StridedSlice(gradient_tape/B/strided_slice_1:output:0.gradient_tape/B/strided_slice_2/stack:output:00gradient_tape/B/strided_slice_2/stack_1:output:00gradient_tape/B/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskr
gradient_tape/B/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ]
gradient_tape/B/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ђ
gradient_tape/B/concatConcatV2&gradient_tape/B/strided_slice:output:0(gradient_tape/B/concat/values_1:output:0(gradient_tape/B/strided_slice_2:output:0$gradient_tape/B/concat/axis:output:0*
N*
T0*
_output_shapes
:X
gradient_tape/B/Size_1Const*
_output_shapes
: *
dtype0*
value	B :X
gradient_tape/B/Size_2Const*
_output_shapes
: *
dtype0*
value	B :]
gradient_tape/B/range/startConst*
_output_shapes
: *
dtype0*
value	B : ]
gradient_tape/B/range/limitConst*
_output_shapes
: *
dtype0*
value	B : ]
gradient_tape/B/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Њ
gradient_tape/B/rangeRange$gradient_tape/B/range/start:output:0$gradient_tape/B/range/limit:output:0$gradient_tape/B/range/delta:output:0*
_output_shapes
: _
gradient_tape/B/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/B/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :­
gradient_tape/B/range_1Range&gradient_tape/B/range_1/start:output:0gradient_tape/B/Size_2:output:0&gradient_tape/B/range_1/delta:output:0*
_output_shapes
:W
gradient_tape/B/add/yConst*
_output_shapes
: *
dtype0*
value	B :~
gradient_tape/B/addAddV2gradient_tape/B/Size_2:output:0gradient_tape/B/add/y:output:0*
T0*
_output_shapes
: _
gradient_tape/B/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/B/range_2Rangegradient_tape/B/add:z:0gradient_tape/B/Size_1:output:0&gradient_tape/B/range_2/delta:output:0*
_output_shapes
: 
gradient_tape/B/ReshapeReshape&gradient_tape/B/ein_YI/Einsum:output:0gradient_tape/B/concat:output:0*
T0*"
_output_shapes
:x
!gradient_tape/B/concat_1/values_1Packgradient_tape/B/Size_2:output:0*
N*
T0*
_output_shapes
:_
gradient_tape/B/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/B/concat_1ConcatV2gradient_tape/B/range:output:0*gradient_tape/B/concat_1/values_1:output:0 gradient_tape/B/range_1:output:0 gradient_tape/B/range_2:output:0&gradient_tape/B/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
gradient_tape/B/transpose	Transpose gradient_tape/B/Reshape:output:0!gradient_tape/B/concat_1:output:0*
T0*"
_output_shapes
:_
gradient_tape/B/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : б
gradient_tape/B/GatherV2GatherV2gradient_tape/B/Cast:y:0!gradient_tape/B/concat_1:output:0&gradient_tape/B/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
gradient_tape/B/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :}
gradient_tape/B/add_1AddV2B/GatherV2_1/axis:output:0 gradient_tape/B/add_1/y:output:0*
T0*
_output_shapes
: w
%gradient_tape/B/strided_slice_3/stackPackB/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:x
'gradient_tape/B/strided_slice_3/stack_1Packgradient_tape/B/add_1:z:0*
N*
T0*
_output_shapes
:q
'gradient_tape/B/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
gradient_tape/B/strided_slice_3StridedSlicegradient_tape/B/Cast:y:0.gradient_tape/B/strided_slice_3/stack:output:00gradient_tape/B/strided_slice_3/stack_1:output:00gradient_tape/B/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
gradient_tape/B/Size_3Const*
_output_shapes
: *
dtype0*
value	B :b
 gradient_tape/B/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/B/ExpandDims_1
ExpandDimsgradient_tape/B/Size_3:output:0)gradient_tape/B/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:
gradient_tape/B/Reshape_1Reshapeb_gatherv2_1_indices%gradient_tape/B/ExpandDims_1:output:0*
T0*
_output_shapes
:т
"gradient_tape/B/UnsortedSegmentSumUnsortedSegmentSumgradient_tape/B/transpose:y:0"gradient_tape/B/Reshape_1:output:0(gradient_tape/B/strided_slice_3:output:0*
Tindices0*
T0*"
_output_shapes
:Y
gradient_tape/B/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/B/add_2AddV2 gradient_tape/B/range_1:output:0 gradient_tape/B/add_2/y:output:0*
T0*
_output_shapes
:k
!gradient_tape/B/concat_2/values_2Const*
_output_shapes
:*
dtype0*
valueB: _
gradient_tape/B/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/B/concat_2ConcatV2gradient_tape/B/range:output:0gradient_tape/B/add_2:z:0*gradient_tape/B/concat_2/values_2:output:0 gradient_tape/B/range_2:output:0&gradient_tape/B/concat_2/axis:output:0*
N*
T0*
_output_shapes
:Ѕ
gradient_tape/B/transpose_1	Transpose+gradient_tape/B/UnsortedSegmentSum:output:0!gradient_tape/B/concat_2:output:0*
T0*"
_output_shapes
:
gradient_tape/B/Shape_1ShapeYI/mul_2:z:0*
T0*
_class
loc:@YI/mul_2*
_output_shapes
:*
out_type0	:эа
gradient_tape/B/Cast_1Cast gradient_tape/B/Shape_1:output:0*

DstT0*

SrcT0	*
_class
loc:@YI/mul_2*
_output_shapes
:X
gradient_tape/B/Size_4Const*
_output_shapes
: *
dtype0*
value	B :b
 gradient_tape/B/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/B/ExpandDims_2
ExpandDimsgradient_tape/B/Size_4:output:0)gradient_tape/B/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:Y
gradient_tape/B/Const_4Const*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/B/Const_5Const*
_output_shapes
: *
dtype0*
value	B :}
%gradient_tape/B/strided_slice_4/stackPack gradient_tape/B/Const_4:output:0*
N*
T0*
_output_shapes
:w
'gradient_tape/B/strided_slice_4/stack_1PackB/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/B/strided_slice_4/stack_2Pack gradient_tape/B/Const_5:output:0*
N*
T0*
_output_shapes
:Ѓ
gradient_tape/B/strided_slice_4StridedSlicegradient_tape/B/Cast_1:y:0.gradient_tape/B/strided_slice_4/stack:output:00gradient_tape/B/strided_slice_4/stack_1:output:00gradient_tape/B/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskY
gradient_tape/B/Const_6Const*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/B/Const_7Const*
_output_shapes
: *
dtype0*
value	B :u
%gradient_tape/B/strided_slice_5/stackPackB/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/B/strided_slice_5/stack_1Pack gradient_tape/B/Const_6:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/B/strided_slice_5/stack_2Pack gradient_tape/B/Const_7:output:0*
N*
T0*
_output_shapes
:Ё
gradient_tape/B/strided_slice_5StridedSlicegradient_tape/B/Cast_1:y:0.gradient_tape/B/strided_slice_5/stack:output:00gradient_tape/B/strided_slice_5/stack_1:output:00gradient_tape/B/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_masko
%gradient_tape/B/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:q
'gradient_tape/B/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'gradient_tape/B/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
gradient_tape/B/strided_slice_6StridedSlice(gradient_tape/B/strided_slice_5:output:0.gradient_tape/B/strided_slice_6/stack:output:00gradient_tape/B/strided_slice_6/stack_1:output:00gradient_tape/B/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskt
!gradient_tape/B/concat_3/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
gradient_tape/B/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
gradient_tape/B/concat_3ConcatV2(gradient_tape/B/strided_slice_4:output:0*gradient_tape/B/concat_3/values_1:output:0(gradient_tape/B/strided_slice_6:output:0&gradient_tape/B/concat_3/axis:output:0*
N*
T0*
_output_shapes
:X
gradient_tape/B/Size_5Const*
_output_shapes
: *
dtype0*
value	B :X
gradient_tape/B/Size_6Const*
_output_shapes
: *
dtype0*
value	B :_
gradient_tape/B/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/B/range_3/limitConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/B/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :В
gradient_tape/B/range_3Range&gradient_tape/B/range_3/start:output:0&gradient_tape/B/range_3/limit:output:0&gradient_tape/B/range_3/delta:output:0*
_output_shapes
: _
gradient_tape/B/range_4/startConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/B/range_4/deltaConst*
_output_shapes
: *
dtype0*
value	B :­
gradient_tape/B/range_4Range&gradient_tape/B/range_4/start:output:0gradient_tape/B/Size_6:output:0&gradient_tape/B/range_4/delta:output:0*
_output_shapes
:Y
gradient_tape/B/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/B/add_3AddV2gradient_tape/B/Size_6:output:0 gradient_tape/B/add_3/y:output:0*
T0*
_output_shapes
: _
gradient_tape/B/range_5/deltaConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/B/range_5Rangegradient_tape/B/add_3:z:0gradient_tape/B/Size_5:output:0&gradient_tape/B/range_5/delta:output:0*
_output_shapes
: А
gradient_tape/B/Reshape_2Reshape(gradient_tape/B/ein_YI/Einsum_1:output:0!gradient_tape/B/concat_3:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџx
!gradient_tape/B/concat_4/values_1Packgradient_tape/B/Size_6:output:0*
N*
T0*
_output_shapes
:_
gradient_tape/B/concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/B/concat_4ConcatV2 gradient_tape/B/range_3:output:0*gradient_tape/B/concat_4/values_1:output:0 gradient_tape/B/range_4:output:0 gradient_tape/B/range_5:output:0&gradient_tape/B/concat_4/axis:output:0*
N*
T0*
_output_shapes
:Ў
gradient_tape/B/transpose_2	Transpose"gradient_tape/B/Reshape_2:output:0!gradient_tape/B/concat_4:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџa
gradient_tape/B/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
gradient_tape/B/GatherV2_1GatherV2gradient_tape/B/Cast_1:y:0!gradient_tape/B/concat_4:output:0(gradient_tape/B/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
gradient_tape/B/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :{
gradient_tape/B/add_4AddV2B/GatherV2/axis:output:0 gradient_tape/B/add_4/y:output:0*
T0*
_output_shapes
: u
%gradient_tape/B/strided_slice_7/stackPackB/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:x
'gradient_tape/B/strided_slice_7/stack_1Packgradient_tape/B/add_4:z:0*
N*
T0*
_output_shapes
:q
'gradient_tape/B/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѕ
gradient_tape/B/strided_slice_7StridedSlicegradient_tape/B/Cast_1:y:0.gradient_tape/B/strided_slice_7/stack:output:00gradient_tape/B/strided_slice_7/stack_1:output:00gradient_tape/B/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
gradient_tape/B/Size_7Const*
_output_shapes
: *
dtype0*
value	B :b
 gradient_tape/B/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/B/ExpandDims_3
ExpandDimsgradient_tape/B/Size_7:output:0)gradient_tape/B/ExpandDims_3/dim:output:0*
T0*
_output_shapes
:
gradient_tape/B/Reshape_3ReshapeB/GatherV2/indices:output:0%gradient_tape/B/ExpandDims_3:output:0*
T0	*
_output_shapes
:ј
$gradient_tape/B/UnsortedSegmentSum_1UnsortedSegmentSumgradient_tape/B/transpose_2:y:0"gradient_tape/B/Reshape_3:output:0(gradient_tape/B/strided_slice_7:output:0*
Tindices0	*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџY
gradient_tape/B/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/B/add_5AddV2 gradient_tape/B/range_4:output:0 gradient_tape/B/add_5/y:output:0*
T0*
_output_shapes
:k
!gradient_tape/B/concat_5/values_2Const*
_output_shapes
:*
dtype0*
valueB: _
gradient_tape/B/concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/B/concat_5ConcatV2 gradient_tape/B/range_3:output:0gradient_tape/B/add_5:z:0*gradient_tape/B/concat_5/values_2:output:0 gradient_tape/B/range_5:output:0&gradient_tape/B/concat_5/axis:output:0*
N*
T0*
_output_shapes
:Й
gradient_tape/B/transpose_3	Transpose-gradient_tape/B/UnsortedSegmentSum_1:output:0!gradient_tape/B/concat_5:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
gradient_tape/YI/mul_2/MulMulgradient_tape/B/transpose_3:y:0
yi_mul_2_y*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ]
gradient_tape/YI/zeros_like	ZerosLikeind_i*
T0*#
_output_shapes
:џџџџџџџџџy
gradient_tape/YI/MaximumMaximumind_igradient_tape/YI/zeros_like:y:0*
T0*#
_output_shapes
:џџџџџџџџџ`
gradient_tape/YI/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ю
gradient_tape/YI/GatherV2GatherV2gradient_tape/YI/mul_2/Mul:z:0gradient_tape/YI/Maximum:z:0'gradient_tape/YI/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџa
gradient_tape/YI/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/YI/GreaterEqualGreaterEqualind_i(gradient_tape/YI/GreaterEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџu
gradient_tape/YI/ShapeShape!gradient_tape/YI/GreaterEqual:z:0*
T0
*
_output_shapes
::эЯW
gradient_tape/YI/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
gradient_tape/YI/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :~
gradient_tape/YI/subSubgradient_tape/YI/Rank:output:0 gradient_tape/YI/Rank_1:output:0*
T0*
_output_shapes
: l
gradient_tape/YI/ones/packedPackgradient_tape/YI/sub:z:0*
N*
T0*
_output_shapes
:]
gradient_tape/YI/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/YI/onesFill%gradient_tape/YI/ones/packed:output:0$gradient_tape/YI/ones/Const:output:0*
T0*
_output_shapes
:^
gradient_tape/YI/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Й
gradient_tape/YI/concatConcatV2gradient_tape/YI/Shape:output:0gradient_tape/YI/ones:output:0%gradient_tape/YI/concat/axis:output:0*
N*
T0*
_output_shapes
:
gradient_tape/YI/ReshapeReshape!gradient_tape/YI/GreaterEqual:z:0 gradient_tape/YI/concat:output:0*
T0
*+
_output_shapes
:џџџџџџџџџ
 gradient_tape/YI/ones_like/ShapeShape"gradient_tape/YI/GatherV2:output:0*
T0*
_output_shapes
::эЯb
 gradient_tape/YI/ones_like/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 ZЗ
gradient_tape/YI/ones_likeFill)gradient_tape/YI/ones_like/Shape:output:0)gradient_tape/YI/ones_like/Const:output:0*
T0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
gradient_tape/YI/and
LogicalAnd!gradient_tape/YI/Reshape:output:0#gradient_tape/YI/ones_like:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
gradient_tape/YI/zeros_like_1	ZerosLike"gradient_tape/YI/GatherV2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџХ
gradient_tape/YI/SelectV2SelectV2gradient_tape/YI/and:z:0"gradient_tape/YI/GatherV2:output:0!gradient_tape/YI/zeros_like_1:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ}
.gradient_tape/YI/trans_120YI/InvertPermutationInvertPermutationYI/trans_120YI/perm:output:0*
_output_shapes
:Ъ
&gradient_tape/YI/trans_120YI/transpose	Transpose"gradient_tape/YI/SelectV2:output:02gradient_tape/YI/trans_120YI/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџg
gradient_tape/YI/zeros_like_2Const*
_output_shapes
:%*
dtype0*
valueB%: 
gradient_tape/YI/Maximum_1Maximumyi_sum_cg_yi_segment_ids&gradient_tape/YI/zeros_like_2:output:0*
T0*
_output_shapes
:%b
 gradient_tape/YI/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
gradient_tape/YI/GatherV2_1GatherV2*gradient_tape/YI/trans_120YI/transpose:y:0gradient_tape/YI/Maximum_1:z:0)gradient_tape/YI/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:%џџџџџџџџџc
!gradient_tape/YI/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/YI/GreaterEqual_1GreaterEqualyi_sum_cg_yi_segment_ids*gradient_tape/YI/GreaterEqual_1/y:output:0*
T0*
_output_shapes
:%b
gradient_tape/YI/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:%Y
gradient_tape/YI/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :Y
gradient_tape/YI/Rank_3Const*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/YI/sub_1Sub gradient_tape/YI/Rank_2:output:0 gradient_tape/YI/Rank_3:output:0*
T0*
_output_shapes
: p
gradient_tape/YI/ones_1/packedPackgradient_tape/YI/sub_1:z:0*
N*
T0*
_output_shapes
:_
gradient_tape/YI/ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/YI/ones_1Fill'gradient_tape/YI/ones_1/packed:output:0&gradient_tape/YI/ones_1/Const:output:0*
T0*
_output_shapes
:`
gradient_tape/YI/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : С
gradient_tape/YI/concat_1ConcatV2!gradient_tape/YI/Shape_1:output:0 gradient_tape/YI/ones_1:output:0'gradient_tape/YI/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
gradient_tape/YI/Reshape_1Reshape#gradient_tape/YI/GreaterEqual_1:z:0"gradient_tape/YI/concat_1:output:0*
T0
*"
_output_shapes
:%
"gradient_tape/YI/ones_like_1/ShapeShape$gradient_tape/YI/GatherV2_1:output:0*
T0*
_output_shapes
::эЯd
"gradient_tape/YI/ones_like_1/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 ZД
gradient_tape/YI/ones_like_1Fill+gradient_tape/YI/ones_like_1/Shape:output:0+gradient_tape/YI/ones_like_1/Const:output:0*
T0
*+
_output_shapes
:%џџџџџџџџџ
gradient_tape/YI/and_1
LogicalAnd#gradient_tape/YI/Reshape_1:output:0%gradient_tape/YI/ones_like_1:output:0*+
_output_shapes
:%џџџџџџџџџ
gradient_tape/YI/zeros_like_3	ZerosLike$gradient_tape/YI/GatherV2_1:output:0*
T0*+
_output_shapes
:%џџџџџџџџџТ
gradient_tape/YI/SelectV2_1SelectV2gradient_tape/YI/and_1:z:0$gradient_tape/YI/GatherV2_1:output:0!gradient_tape/YI/zeros_like_3:y:0*
T0*+
_output_shapes
:%џџџџџџџџџ}
.gradient_tape/YI/trans_201YI/InvertPermutationInvertPermutationYI/trans_201YI/perm:output:0*
_output_shapes
:У
&gradient_tape/YI/trans_201YI/transpose	Transpose$gradient_tape/YI/SelectV2_1:output:02gradient_tape/YI/trans_201YI/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџ%
gradient_tape/YI/mul_1/MulMul*gradient_tape/YI/trans_201YI/transpose:y:0
yi_mul_1_y*
T0*+
_output_shapes
:џџџџџџџџџ%d
gradient_tape/YI/mul_1/ShapeShape
YI/mul:z:0*
T0*
_output_shapes
::эЯs
gradient_tape/YI/mul_1/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"      %   
gradient_tape/YI/mul/MulMulgradient_tape/YI/mul_1/Mul:z:0YI/GatherV2_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ%
gradient_tape/YI/mul/Mul_1MulYI/GatherV2_2:output:0gradient_tape/YI/mul_1/Mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ%n
gradient_tape/YI/mul/ShapeShapeYI/GatherV2_2:output:0*
T0*
_output_shapes
::эЯp
gradient_tape/YI/mul/Shape_1ShapeYI/GatherV2_1:output:0*
T0*
_output_shapes
::эЯУ
*gradient_tape/YI/mul/BroadcastGradientArgsBroadcastGradientArgs#gradient_tape/YI/mul/Shape:output:0%gradient_tape/YI/mul/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЧ
gradient_tape/YI/mul/SumSumgradient_tape/YI/mul/Mul:z:0/gradient_tape/YI/mul/BroadcastGradientArgs:r0:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	keep_dims(Ѕ
gradient_tape/YI/mul/ReshapeReshape!gradient_tape/YI/mul/Sum:output:0#gradient_tape/YI/mul/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ%Ы
gradient_tape/YI/mul/Sum_1Sumgradient_tape/YI/mul/Mul_1:z:0/gradient_tape/YI/mul/BroadcastGradientArgs:r1:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	keep_dims(Ћ
gradient_tape/YI/mul/Reshape_1Reshape#gradient_tape/YI/mul/Sum_1:output:0%gradient_tape/YI/mul/Shape_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ%Є
gradient_tape/YI/Shape_2ShapeYI/einsum/Einsum:output:0*
T0*#
_class
loc:@YI/einsum/Einsum*
_output_shapes
:*
out_type0	:эа
gradient_tape/YI/CastCast!gradient_tape/YI/Shape_2:output:0*

DstT0*

SrcT0	*#
_class
loc:@YI/einsum/Einsum*
_output_shapes
:W
gradient_tape/YI/SizeConst*
_output_shapes
: *
dtype0*
value	B :%a
gradient_tape/YI/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/YI/ExpandDims
ExpandDimsgradient_tape/YI/Size:output:0(gradient_tape/YI/ExpandDims/dim:output:0*
T0*
_output_shapes
:X
gradient_tape/YI/ConstConst*
_output_shapes
: *
dtype0*
value	B : Z
gradient_tape/YI/Const_1Const*
_output_shapes
: *
dtype0*
value	B :{
$gradient_tape/YI/strided_slice/stackPackgradient_tape/YI/Const:output:0*
N*
T0*
_output_shapes
:y
&gradient_tape/YI/strided_slice/stack_1PackYI/GatherV2_2/axis:output:0*
N*
T0*
_output_shapes
:
&gradient_tape/YI/strided_slice/stack_2Pack!gradient_tape/YI/Const_1:output:0*
N*
T0*
_output_shapes
:
gradient_tape/YI/strided_sliceStridedSlicegradient_tape/YI/Cast:y:0-gradient_tape/YI/strided_slice/stack:output:0/gradient_tape/YI/strided_slice/stack_1:output:0/gradient_tape/YI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskZ
gradient_tape/YI/Const_2Const*
_output_shapes
: *
dtype0*
value	B : Z
gradient_tape/YI/Const_3Const*
_output_shapes
: *
dtype0*
value	B :y
&gradient_tape/YI/strided_slice_1/stackPackYI/GatherV2_2/axis:output:0*
N*
T0*
_output_shapes
:
(gradient_tape/YI/strided_slice_1/stack_1Pack!gradient_tape/YI/Const_2:output:0*
N*
T0*
_output_shapes
:
(gradient_tape/YI/strided_slice_1/stack_2Pack!gradient_tape/YI/Const_3:output:0*
N*
T0*
_output_shapes
:Є
 gradient_tape/YI/strided_slice_1StridedSlicegradient_tape/YI/Cast:y:0/gradient_tape/YI/strided_slice_1/stack:output:01gradient_tape/YI/strided_slice_1/stack_1:output:01gradient_tape/YI/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskp
&gradient_tape/YI/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(gradient_tape/YI/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(gradient_tape/YI/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:В
 gradient_tape/YI/strided_slice_2StridedSlice)gradient_tape/YI/strided_slice_1:output:0/gradient_tape/YI/strided_slice_2/stack:output:01gradient_tape/YI/strided_slice_2/stack_1:output:01gradient_tape/YI/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masku
"gradient_tape/YI/concat_2/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ`
gradient_tape/YI/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : §
gradient_tape/YI/concat_2ConcatV2'gradient_tape/YI/strided_slice:output:0+gradient_tape/YI/concat_2/values_1:output:0)gradient_tape/YI/strided_slice_2:output:0'gradient_tape/YI/concat_2/axis:output:0*
N*
T0*
_output_shapes
:Y
gradient_tape/YI/Size_1Const*
_output_shapes
: *
dtype0*
value	B :Y
gradient_tape/YI/Size_2Const*
_output_shapes
: *
dtype0*
value	B :^
gradient_tape/YI/range/startConst*
_output_shapes
: *
dtype0*
value	B : ^
gradient_tape/YI/range/limitConst*
_output_shapes
: *
dtype0*
value	B : ^
gradient_tape/YI/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ў
gradient_tape/YI/rangeRange%gradient_tape/YI/range/start:output:0%gradient_tape/YI/range/limit:output:0%gradient_tape/YI/range/delta:output:0*
_output_shapes
: `
gradient_tape/YI/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : `
gradient_tape/YI/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :Б
gradient_tape/YI/range_1Range'gradient_tape/YI/range_1/start:output:0 gradient_tape/YI/Size_2:output:0'gradient_tape/YI/range_1/delta:output:0*
_output_shapes
:X
gradient_tape/YI/add/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/YI/addAddV2 gradient_tape/YI/Size_2:output:0gradient_tape/YI/add/y:output:0*
T0*
_output_shapes
: `
gradient_tape/YI/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/YI/range_2Rangegradient_tape/YI/add:z:0 gradient_tape/YI/Size_1:output:0'gradient_tape/YI/range_2/delta:output:0*
_output_shapes
: Џ
gradient_tape/YI/Reshape_2Reshape%gradient_tape/YI/mul/Reshape:output:0"gradient_tape/YI/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџz
"gradient_tape/YI/concat_3/values_1Pack gradient_tape/YI/Size_2:output:0*
N*
T0*
_output_shapes
:`
gradient_tape/YI/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/YI/concat_3ConcatV2gradient_tape/YI/range:output:0+gradient_tape/YI/concat_3/values_1:output:0!gradient_tape/YI/range_1:output:0!gradient_tape/YI/range_2:output:0'gradient_tape/YI/concat_3/axis:output:0*
N*
T0*
_output_shapes
:Џ
gradient_tape/YI/transpose	Transpose#gradient_tape/YI/Reshape_2:output:0"gradient_tape/YI/concat_3:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџb
 gradient_tape/YI/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : й
gradient_tape/YI/GatherV2_2GatherV2gradient_tape/YI/Cast:y:0"gradient_tape/YI/concat_3:output:0)gradient_tape/YI/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Z
gradient_tape/YI/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/YI/add_1AddV2YI/GatherV2_2/axis:output:0!gradient_tape/YI/add_1/y:output:0*
T0*
_output_shapes
: y
&gradient_tape/YI/strided_slice_3/stackPackYI/GatherV2_2/axis:output:0*
N*
T0*
_output_shapes
:z
(gradient_tape/YI/strided_slice_3/stack_1Packgradient_tape/YI/add_1:z:0*
N*
T0*
_output_shapes
:r
(gradient_tape/YI/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ј
 gradient_tape/YI/strided_slice_3StridedSlicegradient_tape/YI/Cast:y:0/gradient_tape/YI/strided_slice_3/stack:output:01gradient_tape/YI/strided_slice_3/stack_1:output:01gradient_tape/YI/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
gradient_tape/YI/Size_3Const*
_output_shapes
: *
dtype0*
value	B :%c
!gradient_tape/YI/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/YI/ExpandDims_1
ExpandDims gradient_tape/YI/Size_3:output:0*gradient_tape/YI/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:
gradient_tape/YI/Reshape_3Reshapeyi_gatherv2_2_indices&gradient_tape/YI/ExpandDims_1:output:0*
T0*
_output_shapes
:%ј
#gradient_tape/YI/UnsortedSegmentSumUnsortedSegmentSumgradient_tape/YI/transpose:y:0#gradient_tape/YI/Reshape_3:output:0)gradient_tape/YI/strided_slice_3:output:0*
Tindices0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџZ
gradient_tape/YI/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/YI/add_2AddV2!gradient_tape/YI/range_1:output:0!gradient_tape/YI/add_2/y:output:0*
T0*
_output_shapes
:l
"gradient_tape/YI/concat_4/values_2Const*
_output_shapes
:*
dtype0*
valueB: `
gradient_tape/YI/concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/YI/concat_4ConcatV2gradient_tape/YI/range:output:0gradient_tape/YI/add_2:z:0+gradient_tape/YI/concat_4/values_2:output:0!gradient_tape/YI/range_2:output:0'gradient_tape/YI/concat_4/axis:output:0*
N*
T0*
_output_shapes
:К
gradient_tape/YI/transpose_1	Transpose,gradient_tape/YI/UnsortedSegmentSum:output:0"gradient_tape/YI/concat_4:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
gradient_tape/YI/Shape_3ShapeYI/GatherV2:output:0*
T0*
_class
loc:@YI/GatherV2*
_output_shapes
:*
out_type0	:эа
gradient_tape/YI/Cast_1Cast!gradient_tape/YI/Shape_3:output:0*

DstT0*

SrcT0	*
_class
loc:@YI/GatherV2*
_output_shapes
:Y
gradient_tape/YI/Size_4Const*
_output_shapes
: *
dtype0*
value	B :%c
!gradient_tape/YI/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/YI/ExpandDims_2
ExpandDims gradient_tape/YI/Size_4:output:0*gradient_tape/YI/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:Z
gradient_tape/YI/Const_4Const*
_output_shapes
: *
dtype0*
value	B : Z
gradient_tape/YI/Const_5Const*
_output_shapes
: *
dtype0*
value	B :
&gradient_tape/YI/strided_slice_4/stackPack!gradient_tape/YI/Const_4:output:0*
N*
T0*
_output_shapes
:{
(gradient_tape/YI/strided_slice_4/stack_1PackYI/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:
(gradient_tape/YI/strided_slice_4/stack_2Pack!gradient_tape/YI/Const_5:output:0*
N*
T0*
_output_shapes
:Ј
 gradient_tape/YI/strided_slice_4StridedSlicegradient_tape/YI/Cast_1:y:0/gradient_tape/YI/strided_slice_4/stack:output:01gradient_tape/YI/strided_slice_4/stack_1:output:01gradient_tape/YI/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskZ
gradient_tape/YI/Const_6Const*
_output_shapes
: *
dtype0*
value	B : Z
gradient_tape/YI/Const_7Const*
_output_shapes
: *
dtype0*
value	B :y
&gradient_tape/YI/strided_slice_5/stackPackYI/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:
(gradient_tape/YI/strided_slice_5/stack_1Pack!gradient_tape/YI/Const_6:output:0*
N*
T0*
_output_shapes
:
(gradient_tape/YI/strided_slice_5/stack_2Pack!gradient_tape/YI/Const_7:output:0*
N*
T0*
_output_shapes
:І
 gradient_tape/YI/strided_slice_5StridedSlicegradient_tape/YI/Cast_1:y:0/gradient_tape/YI/strided_slice_5/stack:output:01gradient_tape/YI/strided_slice_5/stack_1:output:01gradient_tape/YI/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskp
&gradient_tape/YI/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(gradient_tape/YI/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(gradient_tape/YI/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:В
 gradient_tape/YI/strided_slice_6StridedSlice)gradient_tape/YI/strided_slice_5:output:0/gradient_tape/YI/strided_slice_6/stack:output:01gradient_tape/YI/strided_slice_6/stack_1:output:01gradient_tape/YI/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masku
"gradient_tape/YI/concat_5/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ`
gradient_tape/YI/concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
gradient_tape/YI/concat_5ConcatV2)gradient_tape/YI/strided_slice_4:output:0+gradient_tape/YI/concat_5/values_1:output:0)gradient_tape/YI/strided_slice_6:output:0'gradient_tape/YI/concat_5/axis:output:0*
N*
T0*
_output_shapes
:Y
gradient_tape/YI/Size_5Const*
_output_shapes
: *
dtype0*
value	B :Y
gradient_tape/YI/Size_6Const*
_output_shapes
: *
dtype0*
value	B :`
gradient_tape/YI/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : `
gradient_tape/YI/range_3/limitConst*
_output_shapes
: *
dtype0*
value	B : `
gradient_tape/YI/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ж
gradient_tape/YI/range_3Range'gradient_tape/YI/range_3/start:output:0'gradient_tape/YI/range_3/limit:output:0'gradient_tape/YI/range_3/delta:output:0*
_output_shapes
: `
gradient_tape/YI/range_4/startConst*
_output_shapes
: *
dtype0*
value	B : `
gradient_tape/YI/range_4/deltaConst*
_output_shapes
: *
dtype0*
value	B :Б
gradient_tape/YI/range_4Range'gradient_tape/YI/range_4/start:output:0 gradient_tape/YI/Size_6:output:0'gradient_tape/YI/range_4/delta:output:0*
_output_shapes
:Z
gradient_tape/YI/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/YI/add_3AddV2 gradient_tape/YI/Size_6:output:0!gradient_tape/YI/add_3/y:output:0*
T0*
_output_shapes
: `
gradient_tape/YI/range_5/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ђ
gradient_tape/YI/range_5Rangegradient_tape/YI/add_3:z:0 gradient_tape/YI/Size_5:output:0'gradient_tape/YI/range_5/delta:output:0*
_output_shapes
: Б
gradient_tape/YI/Reshape_4Reshape'gradient_tape/YI/mul/Reshape_1:output:0"gradient_tape/YI/concat_5:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџz
"gradient_tape/YI/concat_6/values_1Pack gradient_tape/YI/Size_6:output:0*
N*
T0*
_output_shapes
:`
gradient_tape/YI/concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/YI/concat_6ConcatV2!gradient_tape/YI/range_3:output:0+gradient_tape/YI/concat_6/values_1:output:0!gradient_tape/YI/range_4:output:0!gradient_tape/YI/range_5:output:0'gradient_tape/YI/concat_6/axis:output:0*
N*
T0*
_output_shapes
:Б
gradient_tape/YI/transpose_2	Transpose#gradient_tape/YI/Reshape_4:output:0"gradient_tape/YI/concat_6:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџb
 gradient_tape/YI/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : л
gradient_tape/YI/GatherV2_3GatherV2gradient_tape/YI/Cast_1:y:0"gradient_tape/YI/concat_6:output:0)gradient_tape/YI/GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Z
gradient_tape/YI/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/YI/add_4AddV2YI/GatherV2_1/axis:output:0!gradient_tape/YI/add_4/y:output:0*
T0*
_output_shapes
: y
&gradient_tape/YI/strided_slice_7/stackPackYI/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:z
(gradient_tape/YI/strided_slice_7/stack_1Packgradient_tape/YI/add_4:z:0*
N*
T0*
_output_shapes
:r
(gradient_tape/YI/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
 gradient_tape/YI/strided_slice_7StridedSlicegradient_tape/YI/Cast_1:y:0/gradient_tape/YI/strided_slice_7/stack:output:01gradient_tape/YI/strided_slice_7/stack_1:output:01gradient_tape/YI/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
gradient_tape/YI/Size_7Const*
_output_shapes
: *
dtype0*
value	B :%c
!gradient_tape/YI/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/YI/ExpandDims_3
ExpandDims gradient_tape/YI/Size_7:output:0*gradient_tape/YI/ExpandDims_3/dim:output:0*
T0*
_output_shapes
:
gradient_tape/YI/Reshape_5Reshapeyi_gatherv2_1_indices&gradient_tape/YI/ExpandDims_3:output:0*
T0*
_output_shapes
:%ќ
%gradient_tape/YI/UnsortedSegmentSum_1UnsortedSegmentSum gradient_tape/YI/transpose_2:y:0#gradient_tape/YI/Reshape_5:output:0)gradient_tape/YI/strided_slice_7:output:0*
Tindices0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџZ
gradient_tape/YI/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/YI/add_5AddV2!gradient_tape/YI/range_4:output:0!gradient_tape/YI/add_5/y:output:0*
T0*
_output_shapes
:l
"gradient_tape/YI/concat_7/values_2Const*
_output_shapes
:*
dtype0*
valueB: `
gradient_tape/YI/concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/YI/concat_7ConcatV2!gradient_tape/YI/range_3:output:0gradient_tape/YI/add_5:z:0+gradient_tape/YI/concat_7/values_2:output:0!gradient_tape/YI/range_5:output:0'gradient_tape/YI/concat_7/axis:output:0*
N*
T0*
_output_shapes
:М
gradient_tape/YI/transpose_3	Transpose.gradient_tape/YI/UnsortedSegmentSum_1:output:0"gradient_tape/YI/concat_7:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџn
gradient_tape/YI/einsum/ShapeShapeR/GatherV2:output:0*
T0*
_output_shapes
::эЯi
gradient_tape/YI/einsum/Shape_1ShapeY/mul_35:z:0*
T0*
_output_shapes
::эЯВ
gradient_tape/YI/einsum/EinsumEinsum gradient_tape/YI/transpose_1:y:0Y/mul_35:z:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ	*
equationjnl,jl->jnlЗ
 gradient_tape/YI/einsum/Einsum_1Einsum gradient_tape/YI/transpose_1:y:0R/GatherV2:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ	*
equationjnl,jnl->jl
gradient_tape/YI/Shape_4ShapeI/transpose:y:0*
T0*
_class
loc:@I/transpose*
_output_shapes
:*
out_type0	:эа
gradient_tape/YI/Cast_2Cast!gradient_tape/YI/Shape_4:output:0*

DstT0*

SrcT0	*
_class
loc:@I/transpose*
_output_shapes
:G
gradient_tape/YI/Size_8Sizeind_j*
T0*
_output_shapes
: c
!gradient_tape/YI/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/YI/ExpandDims_4
ExpandDims gradient_tape/YI/Size_8:output:0*gradient_tape/YI/ExpandDims_4/dim:output:0*
T0*
_output_shapes
:p
&gradient_tape/YI/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(gradient_tape/YI/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(gradient_tape/YI/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:І
 gradient_tape/YI/strided_slice_8StridedSlicegradient_tape/YI/Cast_2:y:0/gradient_tape/YI/strided_slice_8/stack:output:01gradient_tape/YI/strided_slice_8/stack_1:output:01gradient_tape/YI/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask`
gradient_tape/YI/concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
gradient_tape/YI/concat_8ConcatV2&gradient_tape/YI/ExpandDims_4:output:0)gradient_tape/YI/strided_slice_8:output:0'gradient_tape/YI/concat_8/axis:output:0*
N*
T0*
_output_shapes
:Ё
gradient_tape/YI/Reshape_6Reshape gradient_tape/YI/transpose_3:y:0"gradient_tape/YI/concat_8:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/YI/Reshape_7Reshapeind_j&gradient_tape/YI/ExpandDims_4:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
+gradient_tape/I/transpose/InvertPermutationInvertPermutationI/transpose/perm:output:0*
_output_shapes
:
7gradient_tape/I/transpose/transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9gradient_tape/I/transpose/transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9gradient_tape/I/transpose/transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
1gradient_tape/I/transpose/transpose/strided_sliceStridedSlicegradient_tape/YI/Cast_2:y:0@gradient_tape/I/transpose/transpose/strided_slice/stack:output:0Bgradient_tape/I/transpose/transpose/strided_slice/stack_1:output:0Bgradient_tape/I/transpose/transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%gradient_tape/I/transpose/transpose/xUnsortedSegmentSum#gradient_tape/YI/Reshape_6:output:0#gradient_tape/YI/Reshape_7:output:0:gradient_tape/I/transpose/transpose/strided_slice:output:0*
Tindices0*
T0*+
_output_shapes
:џџџџџџџџџЧ
#gradient_tape/I/transpose/transpose	Transpose.gradient_tape/I/transpose/transpose/x:output:0/gradient_tape/I/transpose/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџЕ
gradient_tape/I/GatherNdGatherNd'gradient_tape/I/transpose/transpose:y:0i_tensorscatteradd_indices*
Tindices0*
Tparams0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/I/IdentityIdentity'gradient_tape/I/transpose/transpose:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradient_tape/I/mul/MulMul!gradient_tape/I/GatherNd:output:0i_mul_y*
T0*+
_output_shapes
:џџџџџџџџџn
gradient_tape/I/ein_A/ShapeShapeI/GatherV2_2:output:0*
T0*
_output_shapes
::эЯn
gradient_tape/I/ein_A/Shape_1ShapeI/GatherV2:output:0*
T0*
_output_shapes
::эЯИ
gradient_tape/I/ein_A/EinsumEinsumgradient_tape/I/mul/Mul:z:0I/GatherV2:output:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ*
equationwak,anw->aknwИ
gradient_tape/I/ein_A/Einsum_1Einsumgradient_tape/I/mul/Mul:z:0I/GatherV2_2:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ*
equationwak,aknw->anw
gradient_tape/I/ShapeConst*
_class
loc:@I/GatherV2_1*
_output_shapes
:*
dtype0	*5
value,B*	"                             
gradient_tape/I/CastCastgradient_tape/I/Shape:output:0*

DstT0*

SrcT0	*
_class
loc:@I/GatherV2_1*
_output_shapes
:J
gradient_tape/I/SizeSizeatomic_mu_i*
T0*
_output_shapes
: `
gradient_tape/I/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/I/ExpandDims
ExpandDimsgradient_tape/I/Size:output:0'gradient_tape/I/ExpandDims/dim:output:0*
T0*
_output_shapes
:m
#gradient_tape/I/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:o
%gradient_tape/I/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%gradient_tape/I/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gradient_tape/I/strided_sliceStridedSlicegradient_tape/I/Cast:y:0,gradient_tape/I/strided_slice/stack:output:0.gradient_tape/I/strided_slice/stack_1:output:0.gradient_tape/I/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask]
gradient_tape/I/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : У
gradient_tape/I/concatConcatV2#gradient_tape/I/ExpandDims:output:0&gradient_tape/I/strided_slice:output:0$gradient_tape/I/concat/axis:output:0*
N*
T0*
_output_shapes
:Є
gradient_tape/I/ReshapeReshape%gradient_tape/I/ein_A/Einsum:output:0gradient_tape/I/concat:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
gradient_tape/I/Reshape_1Reshapeatomic_mu_i#gradient_tape/I/ExpandDims:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
gradient_tape/I/Shape_1Shape	A/mul:z:0*
T0*
_class

loc:@A/mul*
_output_shapes
:*
out_type0	:эа
gradient_tape/I/Cast_1Cast gradient_tape/I/Shape_1:output:0*

DstT0*

SrcT0	*
_class

loc:@A/mul*
_output_shapes
:X
gradient_tape/I/Size_1Const*
_output_shapes
: *
dtype0*
value	B :b
 gradient_tape/I/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/I/ExpandDims_1
ExpandDimsgradient_tape/I/Size_1:output:0)gradient_tape/I/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:W
gradient_tape/I/ConstConst*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/I/Const_1Const*
_output_shapes
: *
dtype0*
value	B :{
%gradient_tape/I/strided_slice_1/stackPackgradient_tape/I/Const:output:0*
N*
T0*
_output_shapes
:w
'gradient_tape/I/strided_slice_1/stack_1PackI/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/I/strided_slice_1/stack_2Pack gradient_tape/I/Const_1:output:0*
N*
T0*
_output_shapes
:Ѓ
gradient_tape/I/strided_slice_1StridedSlicegradient_tape/I/Cast_1:y:0.gradient_tape/I/strided_slice_1/stack:output:00gradient_tape/I/strided_slice_1/stack_1:output:00gradient_tape/I/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskY
gradient_tape/I/Const_2Const*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/I/Const_3Const*
_output_shapes
: *
dtype0*
value	B :u
%gradient_tape/I/strided_slice_2/stackPackI/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/I/strided_slice_2/stack_1Pack gradient_tape/I/Const_2:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/I/strided_slice_2/stack_2Pack gradient_tape/I/Const_3:output:0*
N*
T0*
_output_shapes
:Ё
gradient_tape/I/strided_slice_2StridedSlicegradient_tape/I/Cast_1:y:0.gradient_tape/I/strided_slice_2/stack:output:00gradient_tape/I/strided_slice_2/stack_1:output:00gradient_tape/I/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_masko
%gradient_tape/I/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:q
'gradient_tape/I/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'gradient_tape/I/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
gradient_tape/I/strided_slice_3StridedSlice(gradient_tape/I/strided_slice_2:output:0.gradient_tape/I/strided_slice_3/stack:output:00gradient_tape/I/strided_slice_3/stack_1:output:00gradient_tape/I/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskt
!gradient_tape/I/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
gradient_tape/I/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
gradient_tape/I/concat_1ConcatV2(gradient_tape/I/strided_slice_1:output:0*gradient_tape/I/concat_1/values_1:output:0(gradient_tape/I/strided_slice_3:output:0&gradient_tape/I/concat_1/axis:output:0*
N*
T0*
_output_shapes
:X
gradient_tape/I/Size_2Const*
_output_shapes
: *
dtype0*
value	B :X
gradient_tape/I/Size_3Const*
_output_shapes
: *
dtype0*
value	B :]
gradient_tape/I/range/startConst*
_output_shapes
: *
dtype0*
value	B : ]
gradient_tape/I/range/limitConst*
_output_shapes
: *
dtype0*
value	B : ]
gradient_tape/I/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Њ
gradient_tape/I/rangeRange$gradient_tape/I/range/start:output:0$gradient_tape/I/range/limit:output:0$gradient_tape/I/range/delta:output:0*
_output_shapes
: _
gradient_tape/I/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/I/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :­
gradient_tape/I/range_1Range&gradient_tape/I/range_1/start:output:0gradient_tape/I/Size_3:output:0&gradient_tape/I/range_1/delta:output:0*
_output_shapes
:W
gradient_tape/I/add/yConst*
_output_shapes
: *
dtype0*
value	B :~
gradient_tape/I/addAddV2gradient_tape/I/Size_3:output:0gradient_tape/I/add/y:output:0*
T0*
_output_shapes
: _
gradient_tape/I/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/I/range_2Rangegradient_tape/I/add:z:0gradient_tape/I/Size_2:output:0&gradient_tape/I/range_2/delta:output:0*
_output_shapes
: Џ
gradient_tape/I/Reshape_2Reshape'gradient_tape/I/ein_A/Einsum_1:output:0!gradient_tape/I/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџx
!gradient_tape/I/concat_2/values_1Packgradient_tape/I/Size_3:output:0*
N*
T0*
_output_shapes
:_
gradient_tape/I/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/I/concat_2ConcatV2gradient_tape/I/range:output:0*gradient_tape/I/concat_2/values_1:output:0 gradient_tape/I/range_1:output:0 gradient_tape/I/range_2:output:0&gradient_tape/I/concat_2/axis:output:0*
N*
T0*
_output_shapes
:Ќ
gradient_tape/I/transpose	Transpose"gradient_tape/I/Reshape_2:output:0!gradient_tape/I/concat_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ_
gradient_tape/I/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : г
gradient_tape/I/GatherV2GatherV2gradient_tape/I/Cast_1:y:0!gradient_tape/I/concat_2:output:0&gradient_tape/I/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
gradient_tape/I/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
gradient_tape/I/add_1AddV2I/GatherV2/axis:output:0 gradient_tape/I/add_1/y:output:0*
T0*
_output_shapes
: u
%gradient_tape/I/strided_slice_4/stackPackI/GatherV2/axis:output:0*
N*
T0*
_output_shapes
:x
'gradient_tape/I/strided_slice_4/stack_1Packgradient_tape/I/add_1:z:0*
N*
T0*
_output_shapes
:q
'gradient_tape/I/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѕ
gradient_tape/I/strided_slice_4StridedSlicegradient_tape/I/Cast_1:y:0.gradient_tape/I/strided_slice_4/stack:output:00gradient_tape/I/strided_slice_4/stack_1:output:00gradient_tape/I/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
gradient_tape/I/Size_4Const*
_output_shapes
: *
dtype0*
value	B :b
 gradient_tape/I/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/I/ExpandDims_2
ExpandDimsgradient_tape/I/Size_4:output:0)gradient_tape/I/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:
gradient_tape/I/Reshape_3ReshapeI/GatherV2/indices:output:0%gradient_tape/I/ExpandDims_2:output:0*
T0	*
_output_shapes
:є
"gradient_tape/I/UnsortedSegmentSumUnsortedSegmentSumgradient_tape/I/transpose:y:0"gradient_tape/I/Reshape_3:output:0(gradient_tape/I/strided_slice_4:output:0*
Tindices0	*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџY
gradient_tape/I/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/I/add_2AddV2 gradient_tape/I/range_1:output:0 gradient_tape/I/add_2/y:output:0*
T0*
_output_shapes
:k
!gradient_tape/I/concat_3/values_2Const*
_output_shapes
:*
dtype0*
valueB: _
gradient_tape/I/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/I/concat_3ConcatV2gradient_tape/I/range:output:0gradient_tape/I/add_2:z:0*gradient_tape/I/concat_3/values_2:output:0 gradient_tape/I/range_2:output:0&gradient_tape/I/concat_3/axis:output:0*
N*
T0*
_output_shapes
:З
gradient_tape/I/transpose_1	Transpose+gradient_tape/I/UnsortedSegmentSum:output:0!gradient_tape/I/concat_3:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџА
gradient_tape/I/Shape_2Const*.
_class$
" loc:@I/GatherV2_1/ReadVariableOp*
_output_shapes
:*
dtype0	*5
value,B*	"                             Є
gradient_tape/I/Cast_2Cast gradient_tape/I/Shape_2:output:0*

DstT0*

SrcT0	*.
_class$
" loc:@I/GatherV2_1/ReadVariableOp*
_output_shapes
:X
gradient_tape/I/Size_5Const*
_output_shapes
: *
dtype0*
value	B :b
 gradient_tape/I/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/I/ExpandDims_3
ExpandDimsgradient_tape/I/Size_5:output:0)gradient_tape/I/ExpandDims_3/dim:output:0*
T0*
_output_shapes
:Y
gradient_tape/I/Const_4Const*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/I/Const_5Const*
_output_shapes
: *
dtype0*
value	B :}
%gradient_tape/I/strided_slice_5/stackPack gradient_tape/I/Const_4:output:0*
N*
T0*
_output_shapes
:y
'gradient_tape/I/strided_slice_5/stack_1PackI/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/I/strided_slice_5/stack_2Pack gradient_tape/I/Const_5:output:0*
N*
T0*
_output_shapes
:Ѓ
gradient_tape/I/strided_slice_5StridedSlicegradient_tape/I/Cast_2:y:0.gradient_tape/I/strided_slice_5/stack:output:00gradient_tape/I/strided_slice_5/stack_1:output:00gradient_tape/I/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskY
gradient_tape/I/Const_6Const*
_output_shapes
: *
dtype0*
value	B : Y
gradient_tape/I/Const_7Const*
_output_shapes
: *
dtype0*
value	B :w
%gradient_tape/I/strided_slice_6/stackPackI/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/I/strided_slice_6/stack_1Pack gradient_tape/I/Const_6:output:0*
N*
T0*
_output_shapes
:
'gradient_tape/I/strided_slice_6/stack_2Pack gradient_tape/I/Const_7:output:0*
N*
T0*
_output_shapes
:Ё
gradient_tape/I/strided_slice_6StridedSlicegradient_tape/I/Cast_2:y:0.gradient_tape/I/strided_slice_6/stack:output:00gradient_tape/I/strided_slice_6/stack_1:output:00gradient_tape/I/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_masko
%gradient_tape/I/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:q
'gradient_tape/I/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'gradient_tape/I/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
gradient_tape/I/strided_slice_7StridedSlice(gradient_tape/I/strided_slice_6:output:0.gradient_tape/I/strided_slice_7/stack:output:00gradient_tape/I/strided_slice_7/stack_1:output:00gradient_tape/I/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskt
!gradient_tape/I/concat_4/values_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
gradient_tape/I/concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
gradient_tape/I/concat_4ConcatV2(gradient_tape/I/strided_slice_5:output:0*gradient_tape/I/concat_4/values_1:output:0(gradient_tape/I/strided_slice_7:output:0&gradient_tape/I/concat_4/axis:output:0*
N*
T0*
_output_shapes
:X
gradient_tape/I/Size_6Const*
_output_shapes
: *
dtype0*
value	B :X
gradient_tape/I/Size_7Const*
_output_shapes
: *
dtype0*
value	B :_
gradient_tape/I/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/I/range_3/limitConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/I/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :В
gradient_tape/I/range_3Range&gradient_tape/I/range_3/start:output:0&gradient_tape/I/range_3/limit:output:0&gradient_tape/I/range_3/delta:output:0*
_output_shapes
: _
gradient_tape/I/range_4/startConst*
_output_shapes
: *
dtype0*
value	B : _
gradient_tape/I/range_4/deltaConst*
_output_shapes
: *
dtype0*
value	B :­
gradient_tape/I/range_4Range&gradient_tape/I/range_4/start:output:0gradient_tape/I/Size_7:output:0&gradient_tape/I/range_4/delta:output:0*
_output_shapes
:Y
gradient_tape/I/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/I/add_3AddV2gradient_tape/I/Size_7:output:0 gradient_tape/I/add_3/y:output:0*
T0*
_output_shapes
: _
gradient_tape/I/range_5/deltaConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/I/range_5Rangegradient_tape/I/add_3:z:0gradient_tape/I/Size_6:output:0&gradient_tape/I/range_5/delta:output:0*
_output_shapes
: o
%gradient_tape/I/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'gradient_tape/I/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'gradient_tape/I/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
gradient_tape/I/strided_slice_8StridedSlicegradient_tape/I/Cast:y:0.gradient_tape/I/strided_slice_8/stack:output:00gradient_tape/I/strided_slice_8/stack_1:output:00gradient_tape/I/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskы
$gradient_tape/I/UnsortedSegmentSum_1UnsortedSegmentSum gradient_tape/I/Reshape:output:0"gradient_tape/I/Reshape_1:output:0(gradient_tape/I/strided_slice_8:output:0*
Tindices0*
T0*&
_output_shapes
:Ї
gradient_tape/I/Reshape_4Reshape-gradient_tape/I/UnsortedSegmentSum_1:output:0!gradient_tape/I/concat_4:output:0*
T0*&
_output_shapes
:x
!gradient_tape/I/concat_5/values_1Packgradient_tape/I/Size_7:output:0*
N*
T0*
_output_shapes
:_
gradient_tape/I/concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/I/concat_5ConcatV2 gradient_tape/I/range_3:output:0*gradient_tape/I/concat_5/values_1:output:0 gradient_tape/I/range_4:output:0 gradient_tape/I/range_5:output:0&gradient_tape/I/concat_5/axis:output:0*
N*
T0*
_output_shapes
: 
gradient_tape/I/transpose_2	Transpose"gradient_tape/I/Reshape_4:output:0!gradient_tape/I/concat_5:output:0*
T0*&
_output_shapes
:a
gradient_tape/I/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
gradient_tape/I/GatherV2_1GatherV2gradient_tape/I/Cast_2:y:0!gradient_tape/I/concat_5:output:0(gradient_tape/I/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
gradient_tape/I/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :}
gradient_tape/I/add_4AddV2I/GatherV2_1/axis:output:0 gradient_tape/I/add_4/y:output:0*
T0*
_output_shapes
: w
%gradient_tape/I/strided_slice_9/stackPackI/GatherV2_1/axis:output:0*
N*
T0*
_output_shapes
:x
'gradient_tape/I/strided_slice_9/stack_1Packgradient_tape/I/add_4:z:0*
N*
T0*
_output_shapes
:q
'gradient_tape/I/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѕ
gradient_tape/I/strided_slice_9StridedSlicegradient_tape/I/Cast_2:y:0.gradient_tape/I/strided_slice_9/stack:output:00gradient_tape/I/strided_slice_9/stack_1:output:00gradient_tape/I/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
gradient_tape/I/Size_8Const*
_output_shapes
: *
dtype0*
value	B :b
 gradient_tape/I/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/I/ExpandDims_4
ExpandDimsgradient_tape/I/Size_8:output:0)gradient_tape/I/ExpandDims_4/dim:output:0*
T0*
_output_shapes
:
gradient_tape/I/Reshape_5Reshapei_gatherv2_1_indices%gradient_tape/I/ExpandDims_4:output:0*
T0*
_output_shapes
:ъ
$gradient_tape/I/UnsortedSegmentSum_2UnsortedSegmentSumgradient_tape/I/transpose_2:y:0"gradient_tape/I/Reshape_5:output:0(gradient_tape/I/strided_slice_9:output:0*
Tindices0*
T0*&
_output_shapes
:Y
gradient_tape/I/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :
gradient_tape/I/add_5AddV2 gradient_tape/I/range_4:output:0 gradient_tape/I/add_5/y:output:0*
T0*
_output_shapes
:k
!gradient_tape/I/concat_6/values_2Const*
_output_shapes
:*
dtype0*
valueB: _
gradient_tape/I/concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 
gradient_tape/I/concat_6ConcatV2 gradient_tape/I/range_3:output:0gradient_tape/I/add_5:z:0*gradient_tape/I/concat_6/values_2:output:0 gradient_tape/I/range_5:output:0&gradient_tape/I/concat_6/axis:output:0*
N*
T0*
_output_shapes
:Ћ
gradient_tape/I/transpose_3	Transpose-gradient_tape/I/UnsortedSegmentSum_2:output:0!gradient_tape/I/concat_6:output:0*
T0*&
_output_shapes
:Л
AddN_1AddNgradient_tape/E/transpose_1:y:0!gradient_tape/rho/transpose_1:y:0gradient_tape/I/transpose_1:y:0*
N*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
gradient_tape/A/mul/MulMulAddN_1:sum:0a_mul_y*
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
:џџџџџџџџџ
AddN_2AddN'gradient_tape/YI/einsum/Einsum:output:0&gradient_tape/A/einsum/Einsum:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ	
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
: 
gradient_tape/R/ReshapeReshapeAddN_2:sum:0gradient_tape/R/concat:output:0*
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
 :џџџџџџџџџџџџџџџџџџ
AddN_3AddN)gradient_tape/YI/einsum/Einsum_1:output:0(gradient_tape/A/einsum/Einsum_1:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ	m
gradient_tape/Y/mul_35/MulMulAddN_3:sum:0
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
valueB 2         
mul_9MulDenseLayer/Cast_2:y:0#DenseLayer/einsum_2/Einsum:output:0$^gradient_tape/DenseLayer/mul_11/Mul*
T0*'
_output_shapes
:џџџџџџџџџ@Q
	Sigmoid_1Sigmoid	mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@{
mul_10MulDenseLayer/Cast_2:y:0#DenseLayer/einsum_2/Einsum:output:0*
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
mul_11Mul
mul_10:z:0	sub_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
add_7/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?^
add_7AddV2add_7/x:output:0
mul_11:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_12MulSigmoid_1:y:0	add_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@i
Square_1Square#DenseLayer/einsum_2/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@v
mul_13Mul'gradient_tape/DenseLayer/mul_11/Mul:z:0Square_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_14Mul
mul_13:z:0Sigmoid_1:y:0*
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
mul_15Mul
mul_14:z:0	sub_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       K
Sum_1Sum
mul_15:z:0Const_1:output:0*
T0*
_output_shapes
: t
mul_16Mul'gradient_tape/DenseLayer/mul_11/Mul:z:0
mul_12:z:0*
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
AddN_4AddN9gradient_tape/Y/strided_slice_6/StridedSliceGrad:output:09gradient_tape/Y/strided_slice_8/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_10/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_13/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_12/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_17/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_15/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_14/StridedSliceGrad:output:0:gradient_tape/Y/strided_slice_16/StridedSliceGrad:output:0*
N	*
T0*'
_output_shapes
:џџџџџџџџџЙ
gradient_tape/Y/stack/unstackUnpackAddN_4:sum:0*
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
mul_16:z:0DenseLayer/mul_8:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@*
equation...n,...kn->...kЕ
*gradient_tape/DenseLayer/einsum_2/Einsum_1Einsum
mul_16:z:0DenseLayer/mul_7:z:0*
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
AddN_5AddN'gradient_tape/Y/mul_13/Reshape:output:0'gradient_tape/Y/mul_15/Reshape:output:0'gradient_tape/Y/mul_19/Reshape:output:0)gradient_tape/Y/mul_20/Reshape_1:output:0'gradient_tape/Y/mul_17/Reshape:output:0)gradient_tape/Y/mul_17/Reshape_1:output:0*
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
.gradient_tape/Y/strided_slice/StridedSliceGradStridedSliceGrad,gradient_tape/Y/strided_slice/Shape:output:0=gradient_tape/Y/strided_slice/StridedSliceGrad/begin:output:0;gradient_tape/Y/strided_slice/StridedSliceGrad/end:output:0?gradient_tape/Y/strided_slice/StridedSliceGrad/strides:output:0AddN_5:sum:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskП
AddN_6AddN'gradient_tape/Y/mul_14/Reshape:output:0'gradient_tape/Y/mul_16/Reshape:output:0)gradient_tape/Y/mul_19/Reshape_1:output:0'gradient_tape/Y/mul_20/Reshape:output:0'gradient_tape/Y/mul_18/Reshape:output:0)gradient_tape/Y/mul_18/Reshape_1:output:0*
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
0gradient_tape/Y/strided_slice_1/StridedSliceGradStridedSliceGrad.gradient_tape/Y/strided_slice_1/Shape:output:0?gradient_tape/Y/strided_slice_1/StridedSliceGrad/begin:output:0=gradient_tape/Y/strided_slice_1/StridedSliceGrad/end:output:0Agradient_tape/Y/strided_slice_1/StridedSliceGrad/strides:output:0AddN_6:sum:0*
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
AddN_7AddN&gradient_tape/Y/stack/unstack:output:2)gradient_tape/Y/mul_11/Reshape_1:output:0gradient_tape/Y/mul_12/Mul:z:0*
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
mul_17MulDenseLayer/Cast_1:y:0#DenseLayer/einsum_1/Einsum:output:0#^gradient_tape/DenseLayer/mul_7/Mul*
T0*'
_output_shapes
:џџџџџџџџџ@R
	Sigmoid_2Sigmoid
mul_17:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@{
mul_18MulDenseLayer/Cast_1:y:0#DenseLayer/einsum_1/Einsum:output:0*
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
mul_19Mul
mul_18:z:0	sub_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
add_8/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?^
add_8AddV2add_8/x:output:0
mul_19:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_20MulSigmoid_2:y:0	add_8:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@i
Square_2Square#DenseLayer/einsum_1/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@u
mul_21Mul&gradient_tape/DenseLayer/mul_7/Mul:z:0Square_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_22Mul
mul_21:z:0Sigmoid_2:y:0*
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
mul_23Mul
mul_22:z:0	sub_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@X
Const_2Const*
_output_shapes
:*
dtype0*
valueB"       K
Sum_2Sum
mul_23:z:0Const_2:output:0*
T0*
_output_shapes
: s
mul_24Mul&gradient_tape/DenseLayer/mul_7/Mul:z:0
mul_20:z:0*
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
gradient_tape/Y/mul_6/MulMulAddN_7:sum:0Y/mul_6/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
AddN_8AddN&gradient_tape/Y/stack/unstack:output:1(gradient_tape/Y/mul_7/Reshape_1:output:0*
N*
T0*#
_output_shapes
:џџџџџџџџџj
gradient_tape/Y/mul_5/MulMulY/Sqrt_1:y:0AddN_8:sum:0*
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
AddN_9AddN&gradient_tape/Y/stack/unstack:output:0gradient_tape/Y/mul_8/Mul:z:0*
N*
T0*#
_output_shapes
:џџџџџџџџџp
gradient_tape/Y/mul_2/MulMulAddN_9:sum:0Y/mul_2/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџы
AddN_10AddNgradient_tape/Y/mul_10/Mul:z:0&gradient_tape/Y/mul_7/Reshape:output:0gradient_tape/Y/mul_6/Mul:z:0gradient_tape/Y/mul_5/Mul:z:0gradient_tape/Y/mul_2/Mul:z:0*
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
valueB"      П
0gradient_tape/Y/strided_slice_2/StridedSliceGradStridedSliceGrad.gradient_tape/Y/strided_slice_2/Shape:output:0?gradient_tape/Y/strided_slice_2/StridedSliceGrad/begin:output:0=gradient_tape/Y/strided_slice_2/StridedSliceGrad/end:output:0Agradient_tape/Y/strided_slice_2/StridedSliceGrad/strides:output:0AddN_10:sum:0*
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
mul_24:z:0DenseLayer/mul_4:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@*
equation...n,...kn->...kЕ
*gradient_tape/DenseLayer/einsum_1/Einsum_1Einsum
mul_24:z:0DenseLayer/mul_3:z:0*
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

:@@љ
AddN_11AddN7gradient_tape/Y/strided_slice/StridedSliceGrad:output:09gradient_tape/Y/strided_slice_1/StridedSliceGrad:output:09gradient_tape/Y/strided_slice_2/StridedSliceGrad:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ
.gradient_tape/ScaledBondVector/truediv/RealDivRealDivAddN_11:sum:0ScaledBondVector/add:z:0*
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
:џџџџџџџџџЈ
*gradient_tape/ScaledBondVector/truediv/mulMulAddN_11:sum:04gradient_tape/ScaledBondVector/truediv/RealDiv_2:z:0*
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
mul_25MulDenseLayer/Cast:y:0!DenseLayer/einsum/Einsum:output:0#^gradient_tape/DenseLayer/mul_3/Mul*
T0*'
_output_shapes
:џџџџџџџџџ@R
	Sigmoid_3Sigmoid
mul_25:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
mul_26MulDenseLayer/Cast:y:0!DenseLayer/einsum/Einsum:output:0*
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
mul_27Mul
mul_26:z:0	sub_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
add_9/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?^
add_9AddV2add_9/x:output:0
mul_27:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_28MulSigmoid_3:y:0	add_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@g
Square_3Square!DenseLayer/einsum/Einsum:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@u
mul_29Mul&gradient_tape/DenseLayer/mul_3/Mul:z:0Square_3:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_30Mul
mul_29:z:0Sigmoid_3:y:0*
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
mul_31Mul
mul_30:z:0	sub_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@X
Const_3Const*
_output_shapes
:*
dtype0*
valueB"       K
Sum_3Sum
mul_31:z:0Const_3:output:0*
T0*
_output_shapes
: s
mul_32Mul&gradient_tape/DenseLayer/mul_3/Mul:z:0
mul_28:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
%gradient_tape/DenseLayer/einsum/ShapeShape7SimplifiedBesselRadialBasisFunction/SelectV2_5:output:0*
T0*
_output_shapes
::эЯx
'gradient_tape/DenseLayer/einsum/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   @   Ћ
&gradient_tape/DenseLayer/einsum/EinsumEinsum
mul_32:z:0DenseLayer/mul:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ*
equation...n,...kn->...kж
(gradient_tape/DenseLayer/einsum/Einsum_1Einsum
mul_32:z:07SimplifiedBesselRadialBasisFunction/SelectV2_5:output:0*
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

:@
7gradient_tape/SimplifiedBesselRadialBasisFunction/zerosConst*
_output_shapes
: *
dtype0*
valueB 2        
:gradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2SelectV2/SimplifiedBesselRadialBasisFunction/Greater:z:00gradient_tape/DenseLayer/einsum/Reshape:output:0@gradient_tape/SimplifiedBesselRadialBasisFunction/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
<gradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_1SelectV2/SimplifiedBesselRadialBasisFunction/Greater:z:0@gradient_tape/SimplifiedBesselRadialBasisFunction/zeros:output:00gradient_tape/DenseLayer/einsum/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЇ
7gradient_tape/SimplifiedBesselRadialBasisFunction/ShapeShape2SimplifiedBesselRadialBasisFunction/zeros_like:y:0*
T0*
_output_shapes
::эЯЎ
9gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_1Shape7SimplifiedBesselRadialBasisFunction/SelectV2_5:output:0*
T0*
_output_shapes
::эЯ
Ggradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgsBroadcastGradientArgs@gradient_tape/SimplifiedBesselRadialBasisFunction/Shape:output:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
5gradient_tape/SimplifiedBesselRadialBasisFunction/SumSumCgradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2:output:0Lgradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ј
9gradient_tape/SimplifiedBesselRadialBasisFunction/ReshapeReshape>gradient_tape/SimplifiedBesselRadialBasisFunction/Sum:output:0@gradient_tape/SimplifiedBesselRadialBasisFunction/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџБ
9gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_2Shape:SimplifiedBesselRadialBasisFunction/strided_slice:output:0*
T0*
_output_shapes
::эЯЎ
9gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_3Shape7SimplifiedBesselRadialBasisFunction/SelectV2_5:output:0*
T0*
_output_shapes
::эЯ
Igradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_1BroadcastGradientArgsBgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_2:output:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_3:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЁ
7gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_1SumEgradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_1:output:0Ngradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_1:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ў
;gradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_1Reshape@gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_1:output:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
 gradient_tape/DenseLayer/mul/MulMul2gradient_tape/DenseLayer/einsum/Reshape_1:output:0denselayer_mul_y*
T0*
_output_shapes

:@Д
Egradient_tape/SimplifiedBesselRadialBasisFunction/strided_slice/ShapeShape1SimplifiedBesselRadialBasisFunction/transpose:y:0*
T0*
_output_shapes
::эЯЋ
Vgradient_tape/SimplifiedBesselRadialBasisFunction/strided_slice/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*!
valueB"            Љ
Tgradient_tape/SimplifiedBesselRadialBasisFunction/strided_slice/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*!
valueB"           ­
Xgradient_tape/SimplifiedBesselRadialBasisFunction/strided_slice/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*!
valueB"         
Pgradient_tape/SimplifiedBesselRadialBasisFunction/strided_slice/StridedSliceGradStridedSliceGradNgradient_tape/SimplifiedBesselRadialBasisFunction/strided_slice/Shape:output:0_gradient_tape/SimplifiedBesselRadialBasisFunction/strided_slice/StridedSliceGrad/begin:output:0]gradient_tape/SimplifiedBesselRadialBasisFunction/strided_slice/StridedSliceGrad/end:output:0agradient_tape/SimplifiedBesselRadialBasisFunction/strided_slice/StridedSliceGrad/strides:output:0Dgradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_1:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskЛ
Mgradient_tape/SimplifiedBesselRadialBasisFunction/transpose/InvertPermutationInvertPermutation;SimplifiedBesselRadialBasisFunction/transpose/perm:output:0*
_output_shapes
:Ж
Egradient_tape/SimplifiedBesselRadialBasisFunction/transpose/transpose	TransposeYgradient_tape/SimplifiedBesselRadialBasisFunction/strided_slice/StridedSliceGrad:output:0Qgradient_tape/SimplifiedBesselRadialBasisFunction/transpose/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџф
?gradient_tape/SimplifiedBesselRadialBasisFunction/stack/unstackUnpackIgradient_tape/SimplifiedBesselRadialBasisFunction/transpose/transpose:y:0*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*	
numђ
<gradient_tape/SimplifiedBesselRadialBasisFunction/mul_21/MulMul1SimplifiedBesselRadialBasisFunction/truediv_8:z:0Hgradient_tape/SimplifiedBesselRadialBasisFunction/stack/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ
>gradient_tape/SimplifiedBesselRadialBasisFunction/mul_21/ShapeConst*
_output_shapes
: *
dtype0*
valueB Ќ
@gradient_tape/SimplifiedBesselRadialBasisFunction/mul_21/Shape_1Shape.SimplifiedBesselRadialBasisFunction/add_20:z:0*
T0*
_output_shapes
::эЯЊ
>gradient_tape/SimplifiedBesselRadialBasisFunction/add_20/ShapeShape.SimplifiedBesselRadialBasisFunction/mul_19:z:0*
T0*
_output_shapes
::эЯЌ
@gradient_tape/SimplifiedBesselRadialBasisFunction/add_20/Shape_1Shape.SimplifiedBesselRadialBasisFunction/mul_20:z:0*
T0*
_output_shapes
::эЯЏ
Ngradient_tape/SimplifiedBesselRadialBasisFunction/add_20/BroadcastGradientArgsBroadcastGradientArgsGgradient_tape/SimplifiedBesselRadialBasisFunction/add_20/Shape:output:0Igradient_tape/SimplifiedBesselRadialBasisFunction/add_20/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџІ
<gradient_tape/SimplifiedBesselRadialBasisFunction/add_20/SumSum@gradient_tape/SimplifiedBesselRadialBasisFunction/mul_21/Mul:z:0Sgradient_tape/SimplifiedBesselRadialBasisFunction/add_20/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
@gradient_tape/SimplifiedBesselRadialBasisFunction/add_20/ReshapeReshapeEgradient_tape/SimplifiedBesselRadialBasisFunction/add_20/Sum:output:0Ggradient_tape/SimplifiedBesselRadialBasisFunction/add_20/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
>gradient_tape/SimplifiedBesselRadialBasisFunction/add_20/Sum_1Sum@gradient_tape/SimplifiedBesselRadialBasisFunction/mul_21/Mul:z:0Sgradient_tape/SimplifiedBesselRadialBasisFunction/add_20/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
Bgradient_tape/SimplifiedBesselRadialBasisFunction/add_20/Reshape_1ReshapeGgradient_tape/SimplifiedBesselRadialBasisFunction/add_20/Sum_1:output:0Igradient_tape/SimplifiedBesselRadialBasisFunction/add_20/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџє
<gradient_tape/SimplifiedBesselRadialBasisFunction/mul_19/MulMul2SimplifiedBesselRadialBasisFunction/truediv_10:z:0Igradient_tape/SimplifiedBesselRadialBasisFunction/add_20/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
>gradient_tape/SimplifiedBesselRadialBasisFunction/mul_19/ShapeConst*
_output_shapes
: *
dtype0*
valueB Ќ
@gradient_tape/SimplifiedBesselRadialBasisFunction/mul_19/Shape_1Shape.SimplifiedBesselRadialBasisFunction/add_19:z:0*
T0*
_output_shapes
::эЯђ
<gradient_tape/SimplifiedBesselRadialBasisFunction/mul_20/MulMul.SimplifiedBesselRadialBasisFunction/Sqrt_5:y:0Kgradient_tape/SimplifiedBesselRadialBasisFunction/add_20/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
>gradient_tape/SimplifiedBesselRadialBasisFunction/mul_20/ShapeConst*
_output_shapes
: *
dtype0*
valueB Ћ
@gradient_tape/SimplifiedBesselRadialBasisFunction/mul_20/Shape_1Shape-SimplifiedBesselRadialBasisFunction/mul_8:z:0*
T0*
_output_shapes
::эЯГ
>gradient_tape/SimplifiedBesselRadialBasisFunction/add_19/ShapeShape7SimplifiedBesselRadialBasisFunction/SelectV2_3:output:0*
T0*
_output_shapes
::эЯЕ
@gradient_tape/SimplifiedBesselRadialBasisFunction/add_19/Shape_1Shape7SimplifiedBesselRadialBasisFunction/SelectV2_4:output:0*
T0*
_output_shapes
::эЯЏ
Ngradient_tape/SimplifiedBesselRadialBasisFunction/add_19/BroadcastGradientArgsBroadcastGradientArgsGgradient_tape/SimplifiedBesselRadialBasisFunction/add_19/Shape:output:0Igradient_tape/SimplifiedBesselRadialBasisFunction/add_19/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџІ
<gradient_tape/SimplifiedBesselRadialBasisFunction/add_19/SumSum@gradient_tape/SimplifiedBesselRadialBasisFunction/mul_19/Mul:z:0Sgradient_tape/SimplifiedBesselRadialBasisFunction/add_19/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
@gradient_tape/SimplifiedBesselRadialBasisFunction/add_19/ReshapeReshapeEgradient_tape/SimplifiedBesselRadialBasisFunction/add_19/Sum:output:0Ggradient_tape/SimplifiedBesselRadialBasisFunction/add_19/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
>gradient_tape/SimplifiedBesselRadialBasisFunction/add_19/Sum_1Sum@gradient_tape/SimplifiedBesselRadialBasisFunction/mul_19/Mul:z:0Sgradient_tape/SimplifiedBesselRadialBasisFunction/add_19/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
Bgradient_tape/SimplifiedBesselRadialBasisFunction/add_19/Reshape_1ReshapeGgradient_tape/SimplifiedBesselRadialBasisFunction/add_19/Sum_1:output:0Igradient_tape/SimplifiedBesselRadialBasisFunction/add_19/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџж
AddN_12AddNHgradient_tape/SimplifiedBesselRadialBasisFunction/stack/unstack:output:0@gradient_tape/SimplifiedBesselRadialBasisFunction/mul_20/Mul:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџЖ
;gradient_tape/SimplifiedBesselRadialBasisFunction/mul_8/MulMul1SimplifiedBesselRadialBasisFunction/truediv_1:z:0AddN_12:sum:0*
T0*'
_output_shapes
:џџџџџџџџџ
=gradient_tape/SimplifiedBesselRadialBasisFunction/mul_8/ShapeConst*
_output_shapes
: *
dtype0*
valueB Њ
?gradient_tape/SimplifiedBesselRadialBasisFunction/mul_8/Shape_1Shape-SimplifiedBesselRadialBasisFunction/add_8:z:0*
T0*
_output_shapes
::эЯ
9gradient_tape/SimplifiedBesselRadialBasisFunction/zeros_1Const*
_output_shapes
: *
dtype0*
valueB 2        Н
<gradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_2SelectV22SimplifiedBesselRadialBasisFunction/NotEqual_2:z:0Igradient_tape/SimplifiedBesselRadialBasisFunction/add_19/Reshape:output:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџН
<gradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_3SelectV22SimplifiedBesselRadialBasisFunction/NotEqual_2:z:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/zeros_1:output:0Igradient_tape/SimplifiedBesselRadialBasisFunction/add_19/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЉ
9gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_4Shape2SimplifiedBesselRadialBasisFunction/truediv_12:z:0*
T0*
_output_shapes
::эЯЎ
9gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_5Shape7SimplifiedBesselRadialBasisFunction/SelectV2_3:output:0*
T0*
_output_shapes
::эЯ
Igradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_2BroadcastGradientArgsBgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_4:output:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_5:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЁ
7gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_2SumEgradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_2:output:0Ngradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_2:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ў
;gradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_2Reshape@gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_2:output:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
9gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_6Shape3SimplifiedBesselRadialBasisFunction/ones_like_2:y:0*
T0*
_output_shapes
::эЯЎ
9gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_7Shape7SimplifiedBesselRadialBasisFunction/SelectV2_3:output:0*
T0*
_output_shapes
::эЯ
Igradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_3BroadcastGradientArgsBgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_6:output:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_7:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЁ
7gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_3SumEgradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_3:output:0Ngradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_3:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ў
;gradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_3Reshape@gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_3:output:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
9gradient_tape/SimplifiedBesselRadialBasisFunction/zeros_2Const*
_output_shapes
: *
dtype0*
valueB 2        П
<gradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_4SelectV22SimplifiedBesselRadialBasisFunction/NotEqual_3:z:0Kgradient_tape/SimplifiedBesselRadialBasisFunction/add_19/Reshape_1:output:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/zeros_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџП
<gradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_5SelectV22SimplifiedBesselRadialBasisFunction/NotEqual_3:z:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/zeros_2:output:0Kgradient_tape/SimplifiedBesselRadialBasisFunction/add_19/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџЉ
9gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_8Shape2SimplifiedBesselRadialBasisFunction/truediv_14:z:0*
T0*
_output_shapes
::эЯЎ
9gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_9Shape7SimplifiedBesselRadialBasisFunction/SelectV2_4:output:0*
T0*
_output_shapes
::эЯ
Igradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_4BroadcastGradientArgsBgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_8:output:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_9:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЁ
7gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_4SumEgradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_4:output:0Ngradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_4:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ў
;gradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_4Reshape@gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_4:output:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_8:output:0*
T0*'
_output_shapes
:џџџџџџџџџЋ
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_10Shape3SimplifiedBesselRadialBasisFunction/ones_like_3:y:0*
T0*
_output_shapes
::эЯЏ
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_11Shape7SimplifiedBesselRadialBasisFunction/SelectV2_4:output:0*
T0*
_output_shapes
::эЯ 
Igradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_5BroadcastGradientArgsCgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_10:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_11:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЁ
7gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_5SumEgradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_5:output:0Ngradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_5:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(џ
;gradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_5Reshape@gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_5:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_10:output:0*
T0*'
_output_shapes
:џџџџџџџџџВ
=gradient_tape/SimplifiedBesselRadialBasisFunction/add_8/ShapeShape7SimplifiedBesselRadialBasisFunction/SelectV2_1:output:0*
T0*
_output_shapes
::эЯД
?gradient_tape/SimplifiedBesselRadialBasisFunction/add_8/Shape_1Shape7SimplifiedBesselRadialBasisFunction/SelectV2_2:output:0*
T0*
_output_shapes
::эЯЌ
Mgradient_tape/SimplifiedBesselRadialBasisFunction/add_8/BroadcastGradientArgsBroadcastGradientArgsFgradient_tape/SimplifiedBesselRadialBasisFunction/add_8/Shape:output:0Hgradient_tape/SimplifiedBesselRadialBasisFunction/add_8/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЃ
;gradient_tape/SimplifiedBesselRadialBasisFunction/add_8/SumSum?gradient_tape/SimplifiedBesselRadialBasisFunction/mul_8/Mul:z:0Rgradient_tape/SimplifiedBesselRadialBasisFunction/add_8/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
?gradient_tape/SimplifiedBesselRadialBasisFunction/add_8/ReshapeReshapeDgradient_tape/SimplifiedBesselRadialBasisFunction/add_8/Sum:output:0Fgradient_tape/SimplifiedBesselRadialBasisFunction/add_8/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЅ
=gradient_tape/SimplifiedBesselRadialBasisFunction/add_8/Sum_1Sum?gradient_tape/SimplifiedBesselRadialBasisFunction/mul_8/Mul:z:0Rgradient_tape/SimplifiedBesselRadialBasisFunction/add_8/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
Agradient_tape/SimplifiedBesselRadialBasisFunction/add_8/Reshape_1ReshapeFgradient_tape/SimplifiedBesselRadialBasisFunction/add_8/Sum_1:output:0Hgradient_tape/SimplifiedBesselRadialBasisFunction/add_8/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџћ
Dgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/RealDivRealDivDgradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_2:output:02SimplifiedBesselRadialBasisFunction/truediv_11:z:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
@gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/NegNeg-SimplifiedBesselRadialBasisFunction/Sin_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџ§
Fgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/RealDiv_1RealDivDgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/Neg:y:02SimplifiedBesselRadialBasisFunction/truediv_11:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
Fgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/RealDiv_2RealDivJgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/RealDiv_1:z:02SimplifiedBesselRadialBasisFunction/truediv_11:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
@gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/mulMulDgradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_2:output:0Jgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/RealDiv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ­
Bgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/ShapeShape-SimplifiedBesselRadialBasisFunction/Sin_2:y:0*
T0*
_output_shapes
::эЯД
Dgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/Shape_1Shape2SimplifiedBesselRadialBasisFunction/truediv_11:z:0*
T0*
_output_shapes
::эЯЛ
Rgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/BroadcastGradientArgsBroadcastGradientArgsKgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/Shape:output:0Mgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЖ
@gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/SumSumHgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/RealDiv:z:0Wgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(С
Dgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/ReshapeReshapeIgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/Sum:output:0Kgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/Shape:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџД
Bgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/Sum_1SumDgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/mul:z:0Wgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
Fgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/Reshape_1ReshapeKgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/Sum_1:output:0Mgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџћ
Dgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/RealDivRealDivDgradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_4:output:02SimplifiedBesselRadialBasisFunction/truediv_13:z:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
@gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/NegNeg-SimplifiedBesselRadialBasisFunction/Sin_3:y:0*
T0*'
_output_shapes
:џџџџџџџџџ§
Fgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/RealDiv_1RealDivDgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/Neg:y:02SimplifiedBesselRadialBasisFunction/truediv_13:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
Fgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/RealDiv_2RealDivJgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/RealDiv_1:z:02SimplifiedBesselRadialBasisFunction/truediv_13:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
@gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/mulMulDgradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_4:output:0Jgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/RealDiv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ­
Bgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/ShapeShape-SimplifiedBesselRadialBasisFunction/Sin_3:y:0*
T0*
_output_shapes
::эЯД
Dgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/Shape_1Shape2SimplifiedBesselRadialBasisFunction/truediv_13:z:0*
T0*
_output_shapes
::эЯЛ
Rgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/BroadcastGradientArgsBroadcastGradientArgsKgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/Shape:output:0Mgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЖ
@gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/SumSumHgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/RealDiv:z:0Wgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(С
Dgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/ReshapeReshapeIgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/Sum:output:0Kgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/Shape:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџД
Bgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/Sum_1SumDgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/mul:z:0Wgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
Fgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/Reshape_1ReshapeKgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/Sum_1:output:0Mgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
9gradient_tape/SimplifiedBesselRadialBasisFunction/zeros_3Const*
_output_shapes
: *
dtype0*
valueB 2        К
<gradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_6SelectV20SimplifiedBesselRadialBasisFunction/NotEqual:z:0Hgradient_tape/SimplifiedBesselRadialBasisFunction/add_8/Reshape:output:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/zeros_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџК
<gradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_7SelectV20SimplifiedBesselRadialBasisFunction/NotEqual:z:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/zeros_3:output:0Hgradient_tape/SimplifiedBesselRadialBasisFunction/add_8/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЉ
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_12Shape1SimplifiedBesselRadialBasisFunction/truediv_3:z:0*
T0*
_output_shapes
::эЯЏ
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_13Shape7SimplifiedBesselRadialBasisFunction/SelectV2_1:output:0*
T0*
_output_shapes
::эЯ 
Igradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_6BroadcastGradientArgsCgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_12:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_13:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЁ
7gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_6SumEgradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_6:output:0Ngradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_6:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(џ
;gradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_6Reshape@gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_6:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_12:output:0*
T0*'
_output_shapes
:џџџџџџџџџЉ
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_14Shape1SimplifiedBesselRadialBasisFunction/ones_like:y:0*
T0*
_output_shapes
::эЯЏ
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_15Shape7SimplifiedBesselRadialBasisFunction/SelectV2_1:output:0*
T0*
_output_shapes
::эЯ 
Igradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_7BroadcastGradientArgsCgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_14:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_15:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЁ
7gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_7SumEgradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_7:output:0Ngradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_7:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(џ
;gradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_7Reshape@gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_7:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_14:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
9gradient_tape/SimplifiedBesselRadialBasisFunction/zeros_4Const*
_output_shapes
: *
dtype0*
valueB 2        О
<gradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_8SelectV22SimplifiedBesselRadialBasisFunction/NotEqual_1:z:0Jgradient_tape/SimplifiedBesselRadialBasisFunction/add_8/Reshape_1:output:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/zeros_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџО
<gradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_9SelectV22SimplifiedBesselRadialBasisFunction/NotEqual_1:z:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/zeros_4:output:0Jgradient_tape/SimplifiedBesselRadialBasisFunction/add_8/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџЉ
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_16Shape1SimplifiedBesselRadialBasisFunction/truediv_5:z:0*
T0*
_output_shapes
::эЯЏ
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_17Shape7SimplifiedBesselRadialBasisFunction/SelectV2_2:output:0*
T0*
_output_shapes
::эЯ 
Igradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_8BroadcastGradientArgsCgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_16:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_17:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЁ
7gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_8SumEgradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_8:output:0Ngradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_8:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(џ
;gradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_8Reshape@gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_8:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџЋ
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_18Shape3SimplifiedBesselRadialBasisFunction/ones_like_1:y:0*
T0*
_output_shapes
::эЯЏ
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_19Shape7SimplifiedBesselRadialBasisFunction/SelectV2_2:output:0*
T0*
_output_shapes
::эЯ 
Igradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_9BroadcastGradientArgsCgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_18:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_19:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЁ
7gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_9SumEgradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_9:output:0Ngradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_9:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(џ
;gradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_9Reshape@gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_9:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_18:output:0*
T0*'
_output_shapes
:џџџџџџџџџщ
5gradient_tape/SimplifiedBesselRadialBasisFunction/CosCos2SimplifiedBesselRadialBasisFunction/truediv_11:z:0E^gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/Reshape*
T0*'
_output_shapes
:џџџџџџџџџј
5gradient_tape/SimplifiedBesselRadialBasisFunction/mulMulMgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/Reshape:output:09gradient_tape/SimplifiedBesselRadialBasisFunction/Cos:y:0*
T0*'
_output_shapes
:џџџџџџџџџы
7gradient_tape/SimplifiedBesselRadialBasisFunction/Cos_1Cos2SimplifiedBesselRadialBasisFunction/truediv_13:z:0E^gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/Reshape*
T0*'
_output_shapes
:џџџџџџџџџќ
7gradient_tape/SimplifiedBesselRadialBasisFunction/mul_1MulMgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/Reshape:output:0;gradient_tape/SimplifiedBesselRadialBasisFunction/Cos_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџљ
Cgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/RealDivRealDivDgradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_6:output:01SimplifiedBesselRadialBasisFunction/truediv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџЅ
?gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/NegNeg+SimplifiedBesselRadialBasisFunction/Sin:y:0*
T0*'
_output_shapes
:џџџџџџџџџњ
Egradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/RealDiv_1RealDivCgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/Neg:y:01SimplifiedBesselRadialBasisFunction/truediv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
Egradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/RealDiv_2RealDivIgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/RealDiv_1:z:01SimplifiedBesselRadialBasisFunction/truediv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
?gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/mulMulDgradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_6:output:0Igradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/RealDiv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
Agradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/ShapeShape+SimplifiedBesselRadialBasisFunction/Sin:y:0*
T0*
_output_shapes
::эЯВ
Cgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/Shape_1Shape1SimplifiedBesselRadialBasisFunction/truediv_2:z:0*
T0*
_output_shapes
::эЯИ
Qgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/BroadcastGradientArgsBroadcastGradientArgsJgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/Shape:output:0Lgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџГ
?gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/SumSumGgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/RealDiv:z:0Vgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(О
Cgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/ReshapeReshapeHgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/Sum:output:0Jgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/Shape:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџБ
Agradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/Sum_1SumCgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/mul:z:0Vgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
Egradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/Reshape_1ReshapeJgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/Sum_1:output:0Lgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџљ
Cgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/RealDivRealDivDgradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_8:output:01SimplifiedBesselRadialBasisFunction/truediv_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџЇ
?gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/NegNeg-SimplifiedBesselRadialBasisFunction/Sin_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџњ
Egradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/RealDiv_1RealDivCgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/Neg:y:01SimplifiedBesselRadialBasisFunction/truediv_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
Egradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/RealDiv_2RealDivIgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/RealDiv_1:z:01SimplifiedBesselRadialBasisFunction/truediv_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
?gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/mulMulDgradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_8:output:0Igradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/RealDiv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
Agradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/ShapeShape-SimplifiedBesselRadialBasisFunction/Sin_1:y:0*
T0*
_output_shapes
::эЯВ
Cgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/Shape_1Shape1SimplifiedBesselRadialBasisFunction/truediv_4:z:0*
T0*
_output_shapes
::эЯИ
Qgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/BroadcastGradientArgsBroadcastGradientArgsJgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/Shape:output:0Lgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџГ
?gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/SumSumGgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/RealDiv:z:0Vgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(О
Cgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/ReshapeReshapeHgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/Sum:output:0Jgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/Shape:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџБ
Agradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/Sum_1SumCgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/mul:z:0Vgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
Egradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/Reshape_1ReshapeJgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/Sum_1:output:0Lgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџж
AddN_13AddNOgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_12/Reshape_1:output:09gradient_tape/SimplifiedBesselRadialBasisFunction/mul:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџз
Dgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/RealDivRealDivAddN_13:sum:0ESimplifiedBesselRadialBasisFunction/truediv_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЉ
@gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/NegNeg.SimplifiedBesselRadialBasisFunction/mul_16:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
Fgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/RealDiv_1RealDivDgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/Neg:y:0ESimplifiedBesselRadialBasisFunction/truediv_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Fgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/RealDiv_2RealDivJgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/RealDiv_1:z:0ESimplifiedBesselRadialBasisFunction/truediv_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџд
@gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/mulMulAddN_13:sum:0Jgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/RealDiv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџЎ
Bgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/ShapeShape.SimplifiedBesselRadialBasisFunction/mul_16:z:0*
T0*
_output_shapes
::эЯ
Dgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
Rgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Є
@gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/SumSumDgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/mul:z:0[gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
Dgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/ReshapeReshapeIgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/Sum:output:0Mgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/Shape_1:output:0*
T0*
_output_shapes
: и
AddN_14AddNOgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_14/Reshape_1:output:0;gradient_tape/SimplifiedBesselRadialBasisFunction/mul_1:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџз
Dgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/RealDivRealDivAddN_14:sum:0ESimplifiedBesselRadialBasisFunction/truediv_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЉ
@gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/NegNeg.SimplifiedBesselRadialBasisFunction/mul_18:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
Fgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/RealDiv_1RealDivDgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/Neg:y:0ESimplifiedBesselRadialBasisFunction/truediv_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Fgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/RealDiv_2RealDivJgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/RealDiv_1:z:0ESimplifiedBesselRadialBasisFunction/truediv_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџд
@gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/mulMulAddN_14:sum:0Jgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/RealDiv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџЎ
Bgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/ShapeShape.SimplifiedBesselRadialBasisFunction/mul_18:z:0*
T0*
_output_shapes
::эЯ
Dgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
Rgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Є
@gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/SumSumDgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/mul:z:0[gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
Dgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/ReshapeReshapeIgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/Sum:output:0Mgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/Shape_1:output:0*
T0*
_output_shapes
: щ
7gradient_tape/SimplifiedBesselRadialBasisFunction/Cos_2Cos1SimplifiedBesselRadialBasisFunction/truediv_2:z:0D^gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/Reshape*
T0*'
_output_shapes
:џџџџџџџџџћ
7gradient_tape/SimplifiedBesselRadialBasisFunction/mul_2MulLgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/Reshape:output:0;gradient_tape/SimplifiedBesselRadialBasisFunction/Cos_2:y:0*
T0*'
_output_shapes
:џџџџџџџџџщ
7gradient_tape/SimplifiedBesselRadialBasisFunction/Cos_3Cos1SimplifiedBesselRadialBasisFunction/truediv_4:z:0D^gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/Reshape*
T0*'
_output_shapes
:џџџџџџџџџћ
7gradient_tape/SimplifiedBesselRadialBasisFunction/mul_3MulLgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/Reshape:output:0;gradient_tape/SimplifiedBesselRadialBasisFunction/Cos_3:y:0*
T0*'
_output_shapes
:џџџџџџџџџь
<gradient_tape/SimplifiedBesselRadialBasisFunction/mul_16/MulMulHgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_11/RealDiv:z:0+simplifiedbesselradialbasisfunction_mul_1_y*
T0*'
_output_shapes
:џџџџџџџџџь
<gradient_tape/SimplifiedBesselRadialBasisFunction/mul_18/MulMulHgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_13/RealDiv:z:0+simplifiedbesselradialbasisfunction_mul_1_y*
T0*'
_output_shapes
:џџџџџџџџџз
AddN_15AddNNgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_3/Reshape_1:output:0;gradient_tape/SimplifiedBesselRadialBasisFunction/mul_2:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџе
Cgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/RealDivRealDivAddN_15:sum:0DSimplifiedBesselRadialBasisFunction/truediv_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЇ
?gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/NegNeg-SimplifiedBesselRadialBasisFunction/mul_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
Egradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/RealDiv_1RealDivCgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/Neg:y:0DSimplifiedBesselRadialBasisFunction/truediv_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Egradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/RealDiv_2RealDivIgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/RealDiv_1:z:0DSimplifiedBesselRadialBasisFunction/truediv_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџв
?gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/mulMulAddN_15:sum:0Igradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/RealDiv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
Agradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/ShapeShape-SimplifiedBesselRadialBasisFunction/mul_5:z:0*
T0*
_output_shapes
::эЯ
Cgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
Qgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ё
?gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/SumSumCgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/mul:z:0Zgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
Cgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/ReshapeReshapeHgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/Sum:output:0Lgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/Shape_1:output:0*
T0*
_output_shapes
: з
AddN_16AddNNgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_5/Reshape_1:output:0;gradient_tape/SimplifiedBesselRadialBasisFunction/mul_3:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџе
Cgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/RealDivRealDivAddN_16:sum:0DSimplifiedBesselRadialBasisFunction/truediv_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЇ
?gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/NegNeg-SimplifiedBesselRadialBasisFunction/mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
Egradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/RealDiv_1RealDivCgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/Neg:y:0DSimplifiedBesselRadialBasisFunction/truediv_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Egradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/RealDiv_2RealDivIgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/RealDiv_1:z:0DSimplifiedBesselRadialBasisFunction/truediv_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџв
?gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/mulMulAddN_16:sum:0Igradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/RealDiv_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
Agradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/ShapeShape-SimplifiedBesselRadialBasisFunction/mul_7:z:0*
T0*
_output_shapes
::эЯ
Cgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
Qgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ё
?gradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/SumSumCgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/mul:z:0Zgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
Cgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/ReshapeReshapeHgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/Sum:output:0Lgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/Shape_1:output:0*
T0*
_output_shapes
: ч
<gradient_tape/SimplifiedBesselRadialBasisFunction/mul_15/MulMul@gradient_tape/SimplifiedBesselRadialBasisFunction/mul_16/Mul:z:0.SimplifiedBesselRadialBasisFunction/add_17:z:0*
T0*'
_output_shapes
:џџџџџџџџџч
<gradient_tape/SimplifiedBesselRadialBasisFunction/mul_17/MulMul@gradient_tape/SimplifiedBesselRadialBasisFunction/mul_18/Mul:z:0.SimplifiedBesselRadialBasisFunction/add_18:z:0*
T0*'
_output_shapes
:џџџџџџџџџъ
;gradient_tape/SimplifiedBesselRadialBasisFunction/mul_5/MulMulGgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_2/RealDiv:z:0+simplifiedbesselradialbasisfunction_mul_1_y*
T0*'
_output_shapes
:џџџџџџџџџъ
;gradient_tape/SimplifiedBesselRadialBasisFunction/mul_7/MulMulGgradient_tape/SimplifiedBesselRadialBasisFunction/truediv_4/RealDiv:z:0+simplifiedbesselradialbasisfunction_mul_1_y*
T0*'
_output_shapes
:џџџџџџџџџф
;gradient_tape/SimplifiedBesselRadialBasisFunction/mul_4/MulMul?gradient_tape/SimplifiedBesselRadialBasisFunction/mul_5/Mul:z:0-SimplifiedBesselRadialBasisFunction/add_6:z:0*
T0*'
_output_shapes
:џџџџџџџџџф
;gradient_tape/SimplifiedBesselRadialBasisFunction/mul_6/MulMul?gradient_tape/SimplifiedBesselRadialBasisFunction/mul_7/Mul:z:0-SimplifiedBesselRadialBasisFunction/add_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџа
AddN_17AddN@gradient_tape/SimplifiedBesselRadialBasisFunction/mul_15/Mul:z:0@gradient_tape/SimplifiedBesselRadialBasisFunction/mul_17/Mul:z:0?gradient_tape/SimplifiedBesselRadialBasisFunction/mul_4/Mul:z:0?gradient_tape/SimplifiedBesselRadialBasisFunction/mul_6/Mul:z:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ
9gradient_tape/SimplifiedBesselRadialBasisFunction/zeros_5Const*
_output_shapes
: *
dtype0*
valueB 2        §
=gradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_10SelectV2-SimplifiedBesselRadialBasisFunction/Equal:z:0AddN_17:sum:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/zeros_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ§
=gradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_11SelectV2-SimplifiedBesselRadialBasisFunction/Equal:z:0Bgradient_tape/SimplifiedBesselRadialBasisFunction/zeros_5:output:0AddN_17:sum:0*
T0*'
_output_shapes
:џџџџџџџџџЃ
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_20Shape+SimplifiedBesselRadialBasisFunction/add:z:0*
T0*
_output_shapes
::эЯ­
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_21Shape5SimplifiedBesselRadialBasisFunction/SelectV2:output:0*
T0*
_output_shapes
::эЯЁ
Jgradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_10BroadcastGradientArgsCgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_20:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_21:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЄ
8gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_10SumFgradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_10:output:0Ogradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_10:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
<gradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_10ReshapeAgradient_tape/SimplifiedBesselRadialBasisFunction/Sum_10:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_20:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_22ShapeBondLength/norm/Sqrt:y:0*
T0*
_output_shapes
::эЯ­
:gradient_tape/SimplifiedBesselRadialBasisFunction/Shape_23Shape5SimplifiedBesselRadialBasisFunction/SelectV2:output:0*
T0*
_output_shapes
::эЯЁ
Jgradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_11BroadcastGradientArgsCgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_22:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_23:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџЄ
8gradient_tape/SimplifiedBesselRadialBasisFunction/Sum_11SumFgradient_tape/SimplifiedBesselRadialBasisFunction/SelectV2_11:output:0Ogradient_tape/SimplifiedBesselRadialBasisFunction/BroadcastGradientArgs_11:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
<gradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_11ReshapeAgradient_tape/SimplifiedBesselRadialBasisFunction/Sum_11:output:0Cgradient_tape/SimplifiedBesselRadialBasisFunction/Shape_22:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
AddN_18AddN9gradient_tape/ScaledBondVector/truediv/Reshape_1:output:0Egradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_11:output:0Egradient_tape/SimplifiedBesselRadialBasisFunction/Reshape_10:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ
&gradient_tape/BondLength/norm/SqrtGradSqrtGradBondLength/norm/Sqrt:y:0AddN_18:sum:0*
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
AddN_19AddN7gradient_tape/ScaledBondVector/truediv/Reshape:output:02gradient_tape/BondLength/norm/mul/Reshape:output:04gradient_tape/BondLength/norm/mul/Reshape_1:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџK
NegNegAddN_19:sum:0*
T0*'
_output_shapes
:џџџџџџџџџY
Sum_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : z
Sum_4SumReshape:output:0 Sum_4/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(R
Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB T
Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB `
	Reshape_1Reshapebatch_tot_natReshape_1/shape_1:output:0*
T0*
_output_shapes
: 
UnsortedSegmentSumUnsortedSegmentSumNeg:y:0ind_jReshape_1:output:0*
Tindices0*
T0*'
_output_shapes
:џџџџџџџџџ
UnsortedSegmentSum_1UnsortedSegmentSumNeg:y:0ind_iReshape_1:output:0*
Tindices0*
T0*'
_output_shapes
:џџџџџџџџџz
sub_8SubUnsortedSegmentSum:output:0UnsortedSegmentSum_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџU
mul_33MulNeg:y:0bond_vector*
T0*'
_output_shapes
:џџџџџџџџџY
Sum_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : _
Sum_5Sum
mul_33:z:0 Sum_5/reduction_indices:output:0*
T0*
_output_shapes
:b
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџc
	Reshape_2ReshapeSum_5:output:0Reshape_2/shape:output:0*
T0*
_output_shapes
:f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_4StridedSliceNeg:y:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_5StridedSlicebond_vectorstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_masko
mul_34Mulstrided_slice_4:output:0strided_slice_5:output:0*
T0*#
_output_shapes
:џџџџџџџџџY
Sum_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : [
Sum_6Sum
mul_34:z:0 Sum_6/reduction_indices:output:0*
T0*
_output_shapes
: Y
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:c
	Reshape_3ReshapeSum_6:output:0Reshape_3/shape:output:0*
T0*
_output_shapes
:f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_6StridedSliceNeg:y:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_7StridedSlicebond_vectorstrided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_masko
mul_35Mulstrided_slice_6:output:0strided_slice_7:output:0*
T0*#
_output_shapes
:џџџџџџџџџY
Sum_7/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : [
Sum_7Sum
mul_35:z:0 Sum_7/reduction_indices:output:0*
T0*
_output_shapes
: Y
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:c
	Reshape_4ReshapeSum_7:output:0Reshape_4/shape:output:0*
T0*
_output_shapes
:f
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_8StridedSliceNeg:y:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_9StridedSlicebond_vectorstrided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_masko
mul_36Mulstrided_slice_8:output:0strided_slice_9:output:0*
T0*#
_output_shapes
:џџџџџџџџџY
Sum_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : [
Sum_8Sum
mul_36:z:0 Sum_8/reduction_indices:output:0*
T0*
_output_shapes
: Y
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:c
	Reshape_5ReshapeSum_8:output:0Reshape_5/shape:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : І
concatConcatV2Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:_
IdentityIdentityReshape:output:0^NoOp*
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
:X

Identity_4IdentityNeg:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЮ
NoOpNoOp^B/GatherV2_1/ReadVariableOp^ChemIndTransf/ReadVariableOp+^ChemIndTransf/einsum/Einsum/ReadVariableOp^DenseLayer/ReadVariableOp^DenseLayer/ReadVariableOp_1^DenseLayer/ReadVariableOp_2^DenseLayer/ReadVariableOp_3^DenseLayer/ReadVariableOp_4^DenseLayer/ReadVariableOp_5^E/GatherV2_1/ReadVariableOp^E2/GatherV2_1/ReadVariableOp^I/GatherV2_1/ReadVariableOp9^SimplifiedBesselRadialBasisFunction/Pow_1/ReadVariableOp9^SimplifiedBesselRadialBasisFunction/Pow_8/ReadVariableOp>^SimplifiedBesselRadialBasisFunction/truediv_11/ReadVariableOp>^SimplifiedBesselRadialBasisFunction/truediv_13/ReadVariableOp=^SimplifiedBesselRadialBasisFunction/truediv_2/ReadVariableOp=^SimplifiedBesselRadialBasisFunction/truediv_4/ReadVariableOp^rho/GatherV2_1/ReadVariableOp^rho2/GatherV2_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
_XlaMustCompile( *(
_construction_contextkEagerRuntime*е
_input_shapesУ
Р:џџџџџџџџџ: : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : :	: ::: : : : : : :: :: :: :: :: ::%:%:%:%: : : :: :: :: :: :: :: : : : : : 2:
B/GatherV2_1/ReadVariableOpB/GatherV2_1/ReadVariableOp2<
ChemIndTransf/ReadVariableOpChemIndTransf/ReadVariableOp2X
*ChemIndTransf/einsum/Einsum/ReadVariableOp*ChemIndTransf/einsum/Einsum/ReadVariableOp2:
DenseLayer/ReadVariableOp_1DenseLayer/ReadVariableOp_12:
DenseLayer/ReadVariableOp_2DenseLayer/ReadVariableOp_22:
DenseLayer/ReadVariableOp_3DenseLayer/ReadVariableOp_32:
DenseLayer/ReadVariableOp_4DenseLayer/ReadVariableOp_42:
DenseLayer/ReadVariableOp_5DenseLayer/ReadVariableOp_526
DenseLayer/ReadVariableOpDenseLayer/ReadVariableOp2:
E/GatherV2_1/ReadVariableOpE/GatherV2_1/ReadVariableOp2<
E2/GatherV2_1/ReadVariableOpE2/GatherV2_1/ReadVariableOp2:
I/GatherV2_1/ReadVariableOpI/GatherV2_1/ReadVariableOp2t
8SimplifiedBesselRadialBasisFunction/Pow_1/ReadVariableOp8SimplifiedBesselRadialBasisFunction/Pow_1/ReadVariableOp2t
8SimplifiedBesselRadialBasisFunction/Pow_8/ReadVariableOp8SimplifiedBesselRadialBasisFunction/Pow_8/ReadVariableOp2~
=SimplifiedBesselRadialBasisFunction/truediv_11/ReadVariableOp=SimplifiedBesselRadialBasisFunction/truediv_11/ReadVariableOp2~
=SimplifiedBesselRadialBasisFunction/truediv_13/ReadVariableOp=SimplifiedBesselRadialBasisFunction/truediv_13/ReadVariableOp2|
<SimplifiedBesselRadialBasisFunction/truediv_2/ReadVariableOp<SimplifiedBesselRadialBasisFunction/truediv_2/ReadVariableOp2|
<SimplifiedBesselRadialBasisFunction/truediv_4/ReadVariableOp<SimplifiedBesselRadialBasisFunction/truediv_4/ReadVariableOp2>
rho/GatherV2_1/ReadVariableOprho/GatherV2_1/ReadVariableOp2@
rho2/GatherV2_1/ReadVariableOprho2/GatherV2_1/ReadVariableOp:@

_output_shapes
: :?

_output_shapes
: :(>$
"
_user_specified_name
resource:=

_output_shapes
: :(<$
"
_user_specified_name
resource:;

_output_shapes
: :$: 

_output_shapes

::9

_output_shapes
: : 8

_output_shapes
::(7$
"
_user_specified_name
resource:$6 

_output_shapes

::5

_output_shapes
: : 4

_output_shapes
::(3$
"
_user_specified_name
resource:$2 

_output_shapes

::1

_output_shapes
: : 0

_output_shapes
::(/$
"
_user_specified_name
resource:.

_output_shapes
: :-

_output_shapes
: : ,

_output_shapes
:%:(+$
"
_output_shapes
:%: *

_output_shapes
:%: )

_output_shapes
:%:$( 

_output_shapes

::'

_output_shapes
: : &

_output_shapes
::(%$
"
_user_specified_name
resource:$$ 

_output_shapes

::#

_output_shapes
: : "

_output_shapes
::(!$
"
_user_specified_name
resource:$  

_output_shapes

::

_output_shapes
: : 

_output_shapes
::($
"
_user_specified_name
resource:

_output_shapes
: :($
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
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
:	:

_output_shapes
: :($
"
_user_specified_name
resource:
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
resource:($
"
_user_specified_name
resource:


_output_shapes
: :	

_output_shapes
: :
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
Ќ

__inference__traced_save_143775
file_prefix8
*read_disablecopyonread_element_map_symbols:8
*read_1_disablecopyonread_element_map_index:>
,read_2_disablecopyonread_z_chemicalembedding:?
%read_3_disablecopyonread_i_reducing_a:?
%read_4_disablecopyonread_e_reducing_a:A
'read_5_disablecopyonread_rho_reducing_a:<
&read_6_disablecopyonread_b_reducing_yi:@
&read_7_disablecopyonread_e2_reducing_b:B
(read_8_disablecopyonread_rho2_reducing_b:)
read_9_disablecopyonread_cutoff: T
Bread_10_disablecopyonread_chemindtransf_denselayer_chemindtransf__:W
Eread_11_disablecopyonread_denselayer_denselayer_denselayer_no_decay_3:@W
Eread_12_disablecopyonread_denselayer_denselayer_denselayer_no_decay_2:@@W
Eread_13_disablecopyonread_denselayer_denselayer_denselayer_no_decay_1:@@U
Cread_14_disablecopyonread_denselayer_denselayer_denselayer_no_decay:@P
>read_15_disablecopyonread_denselayer_denselayer_denselayer___1: N
<read_16_disablecopyonread_denselayer_denselayer_denselayer__: 
savev2_const_42
identity_35ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
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
: |
Read/DisableCopyOnReadDisableCopyOnRead*read_disablecopyonread_element_map_symbols"/device:CPU:0*
_output_shapes
 Ђ
Read/ReadVariableOpReadVariableOp*read_disablecopyonread_element_map_symbols^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0e
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_1/DisableCopyOnReadDisableCopyOnRead*read_1_disablecopyonread_element_map_index"/device:CPU:0*
_output_shapes
 І
Read_1/ReadVariableOpReadVariableOp*read_1_disablecopyonread_element_map_index^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_2/DisableCopyOnReadDisableCopyOnRead,read_2_disablecopyonread_z_chemicalembedding"/device:CPU:0*
_output_shapes
 Ќ
Read_2/ReadVariableOpReadVariableOp,read_2_disablecopyonread_z_chemicalembedding^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_i_reducing_a"/device:CPU:0*
_output_shapes
 ­
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_i_reducing_a^Read_3/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*&
_output_shapes
:y
Read_4/DisableCopyOnReadDisableCopyOnRead%read_4_disablecopyonread_e_reducing_a"/device:CPU:0*
_output_shapes
 ­
Read_4/ReadVariableOpReadVariableOp%read_4_disablecopyonread_e_reducing_a^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:{
Read_5/DisableCopyOnReadDisableCopyOnRead'read_5_disablecopyonread_rho_reducing_a"/device:CPU:0*
_output_shapes
 Џ
Read_5/ReadVariableOpReadVariableOp'read_5_disablecopyonread_rho_reducing_a^Read_5/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0v
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_b_reducing_yi"/device:CPU:0*
_output_shapes
 Њ
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_b_reducing_yi^Read_6/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0r
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*"
_output_shapes
:z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_e2_reducing_b"/device:CPU:0*
_output_shapes
 Ў
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_e2_reducing_b^Read_7/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0v
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*&
_output_shapes
:|
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_rho2_reducing_b"/device:CPU:0*
_output_shapes
 А
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_rho2_reducing_b^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
:s
Read_9/DisableCopyOnReadDisableCopyOnReadread_9_disablecopyonread_cutoff"/device:CPU:0*
_output_shapes
 
Read_9/ReadVariableOpReadVariableOpread_9_disablecopyonread_cutoff^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_10/DisableCopyOnReadDisableCopyOnReadBread_10_disablecopyonread_chemindtransf_denselayer_chemindtransf__"/device:CPU:0*
_output_shapes
 Ф
Read_10/ReadVariableOpReadVariableOpBread_10_disablecopyonread_chemindtransf_denselayer_chemindtransf__^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_11/DisableCopyOnReadDisableCopyOnReadEread_11_disablecopyonread_denselayer_denselayer_denselayer_no_decay_3"/device:CPU:0*
_output_shapes
 Ч
Read_11/ReadVariableOpReadVariableOpEread_11_disablecopyonread_denselayer_denselayer_denselayer_no_decay_3^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_12/DisableCopyOnReadDisableCopyOnReadEread_12_disablecopyonread_denselayer_denselayer_denselayer_no_decay_2"/device:CPU:0*
_output_shapes
 Ч
Read_12/ReadVariableOpReadVariableOpEread_12_disablecopyonread_denselayer_denselayer_denselayer_no_decay_2^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:@@
Read_13/DisableCopyOnReadDisableCopyOnReadEread_13_disablecopyonread_denselayer_denselayer_denselayer_no_decay_1"/device:CPU:0*
_output_shapes
 Ч
Read_13/ReadVariableOpReadVariableOpEread_13_disablecopyonread_denselayer_denselayer_denselayer_no_decay_1^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:@@
Read_14/DisableCopyOnReadDisableCopyOnReadCread_14_disablecopyonread_denselayer_denselayer_denselayer_no_decay"/device:CPU:0*
_output_shapes
 Х
Read_14/ReadVariableOpReadVariableOpCread_14_disablecopyonread_denselayer_denselayer_denselayer_no_decay^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_15/DisableCopyOnReadDisableCopyOnRead>read_15_disablecopyonread_denselayer_denselayer_denselayer___1"/device:CPU:0*
_output_shapes
 Р
Read_15/ReadVariableOpReadVariableOp>read_15_disablecopyonread_denselayer_denselayer_denselayer___1^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

: 
Read_16/DisableCopyOnReadDisableCopyOnRead<read_16_disablecopyonread_denselayer_denselayer_denselayer__"/device:CPU:0*
_output_shapes
 О
Read_16/ReadVariableOpReadVariableOp<read_16_disablecopyonread_denselayer_denselayer_denselayer__^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

: Х
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ю
valueфBсB=instructions/5/element_map_symbols/.ATTRIBUTES/VARIABLE_VALUEB;instructions/5/element_map_index/.ATTRIBUTES/VARIABLE_VALUEB+instructions/5/w/.ATTRIBUTES/VARIABLE_VALUEB4instructions/7/reducing_A/.ATTRIBUTES/VARIABLE_VALUEB4instructions/8/reducing_A/.ATTRIBUTES/VARIABLE_VALUEB4instructions/9/reducing_A/.ATTRIBUTES/VARIABLE_VALUEB6instructions/11/reducing_YI/.ATTRIBUTES/VARIABLE_VALUEB5instructions/12/reducing_B/.ATTRIBUTES/VARIABLE_VALUEB5instructions/13/reducing_B/.ATTRIBUTES/VARIABLE_VALUEB;instructions/2/basis_function/rc/.ATTRIBUTES/VARIABLE_VALUEB9instructions/6/lin_transform/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer0/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer1/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer2/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer3/w/.ATTRIBUTES/VARIABLE_VALUEB7instructions/16/mlp/layer0/w/.ATTRIBUTES/VARIABLE_VALUEB7instructions/16/mlp/layer1/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B н
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0savev2_const_42"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 * 
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_34Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_35IdentityIdentity_34:output:0^NoOp*
T0*
_output_shapes
: Є
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_35Identity_35:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:@<

_output_shapes
: 
"
_user_specified_name
Const_42:B>
<
_user_specified_name$"DenseLayer/DenseLayer_DenseLayer__:D@
>
_user_specified_name&$DenseLayer/DenseLayer_DenseLayer___1:IE
C
_user_specified_name+)DenseLayer/DenseLayer_DenseLayer_no_decay:KG
E
_user_specified_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_1:KG
E
_user_specified_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_2:KG
E
_user_specified_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_3:HD
B
_user_specified_name*(ChemIndTransf/DenseLayer_ChemIndTransf__:&
"
 
_user_specified_namecutoff:/	+
)
_user_specified_namerho2/reducing_B:-)
'
_user_specified_nameE2/reducing_B:-)
'
_user_specified_nameB/reducing_YI:.*
(
_user_specified_namerho/reducing_A:,(
&
_user_specified_nameE/reducing_A:,(
&
_user_specified_nameI/reducing_A:3/
-
_user_specified_nameZ/ChemicalEmbedding:1-
+
_user_specified_nameelement_map_index:3/
-
_user_specified_nameelement_map_symbols:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
V

"__inference__traced_restore_143835
file_prefix2
$assignvariableop_element_map_symbols:2
$assignvariableop_1_element_map_index:8
&assignvariableop_2_z_chemicalembedding:9
assignvariableop_3_i_reducing_a:9
assignvariableop_4_e_reducing_a:;
!assignvariableop_5_rho_reducing_a:6
 assignvariableop_6_b_reducing_yi::
 assignvariableop_7_e2_reducing_b:<
"assignvariableop_8_rho2_reducing_b:#
assignvariableop_9_cutoff: N
<assignvariableop_10_chemindtransf_denselayer_chemindtransf__:Q
?assignvariableop_11_denselayer_denselayer_denselayer_no_decay_3:@Q
?assignvariableop_12_denselayer_denselayer_denselayer_no_decay_2:@@Q
?assignvariableop_13_denselayer_denselayer_denselayer_no_decay_1:@@O
=assignvariableop_14_denselayer_denselayer_denselayer_no_decay:@J
8assignvariableop_15_denselayer_denselayer_denselayer___1: H
6assignvariableop_16_denselayer_denselayer_denselayer__: 
identity_18ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ш
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ю
valueфBсB=instructions/5/element_map_symbols/.ATTRIBUTES/VARIABLE_VALUEB;instructions/5/element_map_index/.ATTRIBUTES/VARIABLE_VALUEB+instructions/5/w/.ATTRIBUTES/VARIABLE_VALUEB4instructions/7/reducing_A/.ATTRIBUTES/VARIABLE_VALUEB4instructions/8/reducing_A/.ATTRIBUTES/VARIABLE_VALUEB4instructions/9/reducing_A/.ATTRIBUTES/VARIABLE_VALUEB6instructions/11/reducing_YI/.ATTRIBUTES/VARIABLE_VALUEB5instructions/12/reducing_B/.ATTRIBUTES/VARIABLE_VALUEB5instructions/13/reducing_B/.ATTRIBUTES/VARIABLE_VALUEB;instructions/2/basis_function/rc/.ATTRIBUTES/VARIABLE_VALUEB9instructions/6/lin_transform/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer0/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer1/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer2/w/.ATTRIBUTES/VARIABLE_VALUEB6instructions/3/mlp/layer3/w/.ATTRIBUTES/VARIABLE_VALUEB7instructions/16/mlp/layer0/w/.ATTRIBUTES/VARIABLE_VALUEB7instructions/16/mlp/layer1/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B ј
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOpAssignVariableOp$assignvariableop_element_map_symbolsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_1AssignVariableOp$assignvariableop_1_element_map_indexIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_2AssignVariableOp&assignvariableop_2_z_chemicalembeddingIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_3AssignVariableOpassignvariableop_3_i_reducing_aIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_4AssignVariableOpassignvariableop_4_e_reducing_aIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_5AssignVariableOp!assignvariableop_5_rho_reducing_aIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_6AssignVariableOp assignvariableop_6_b_reducing_yiIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_7AssignVariableOp assignvariableop_7_e2_reducing_bIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_8AssignVariableOp"assignvariableop_8_rho2_reducing_bIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_9AssignVariableOpassignvariableop_9_cutoffIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_10AssignVariableOp<assignvariableop_10_chemindtransf_denselayer_chemindtransf__Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_11AssignVariableOp?assignvariableop_11_denselayer_denselayer_denselayer_no_decay_3Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_12AssignVariableOp?assignvariableop_12_denselayer_denselayer_denselayer_no_decay_2Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_13AssignVariableOp?assignvariableop_13_denselayer_denselayer_denselayer_no_decay_1Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_14AssignVariableOp=assignvariableop_14_denselayer_denselayer_denselayer_no_decayIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_15AssignVariableOp8assignvariableop_15_denselayer_denselayer_denselayer___1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_16AssignVariableOp6assignvariableop_16_denselayer_denselayer_denselayer__Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Х
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_18IdentityIdentity_17:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_18Identity_18:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$: : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:B>
<
_user_specified_name$"DenseLayer/DenseLayer_DenseLayer__:D@
>
_user_specified_name&$DenseLayer/DenseLayer_DenseLayer___1:IE
C
_user_specified_name+)DenseLayer/DenseLayer_DenseLayer_no_decay:KG
E
_user_specified_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_1:KG
E
_user_specified_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_2:KG
E
_user_specified_name-+DenseLayer/DenseLayer_DenseLayer_no_decay_3:HD
B
_user_specified_name*(ChemIndTransf/DenseLayer_ChemIndTransf__:&
"
 
_user_specified_namecutoff:/	+
)
_user_specified_namerho2/reducing_B:-)
'
_user_specified_nameE2/reducing_B:-)
'
_user_specified_nameB/reducing_YI:.*
(
_user_specified_namerho/reducing_A:,(
&
_user_specified_nameE/reducing_A:,(
&
_user_specified_nameI/reducing_A:3/
-
_user_specified_nameZ/ChemicalEmbedding:1-
+
_user_specified_nameelement_map_index:3/
-
_user_specified_nameelement_map_symbols:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
в
О
#__inference_internal_grad_fn_143634
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
в
О
#__inference_internal_grad_fn_143688
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
:џџџџџџџџџ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ u
mul_1Mulmul_denselayer_cast_3mul_denselayer_einsum_4_einsum*
T0*'
_output_shapes
:џџџџџџџџџ N
sub/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ N
add/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
SquareSquaremul_denselayer_einsum_4_einsum*
T0*'
_output_shapes
:џџџџџџџџџ Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ P
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      №?]
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ V
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
:џџџџџџџџџ Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ :џџџџџџџџџ : : :џџџџџџџџџ :c_
'
_output_shapes
:џџџџџџџџџ 
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
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0<
#__inference_internal_grad_fn_143607CustomGradient-140728<
#__inference_internal_grad_fn_143634CustomGradient-140745<
#__inference_internal_grad_fn_143661CustomGradient-140762<
#__inference_internal_grad_fn_143688CustomGradient-141224"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*І
serving_default
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
StatefulPartitionedCall:3<
z_pair_f0
StatefulPartitionedCall:4џџџџџџџџџtensorflow/serving/predict:а
q
instructions
compute_specs
train_specs
compute

signatures"
_generic_user_object
І
0
1
2
	3

4
5
6
7
8
9
10
11
12
13
14
15
16
17"
trackable_list_wrapper

	ind_i
	ind_j
bond_vector
mu_i
mu_j
atomic_mu_i
batch_tot_nat
batch_tot_nat_real"
trackable_dict_wrapper
Ч
	 ind_i
	!ind_j
"map_atoms_to_structure
#n_struct_total
$bond_vector
%batch_tot_nat
&mu_i
'mu_j
(atomic_mu_i
)batch_tot_nat_real"
trackable_dict_wrapper
М
*trace_02
__inference_compute_143355
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
mu_jџџџџџџџџџ0z*trace_0
,
+serving_default"
signature_map
.
,
_init_args"
_generic_user_object
.
-
_init_args"
_generic_user_object
N
.
_init_args

/kwargs
0basis_function"
_generic_user_object
J
1
_init_args
2hidden_layers
3mlp"
_generic_user_object
6
4
_init_args
5sg"
_generic_user_object
e
6
_init_args
7element_map_symbols
8element_map_index
9w"
_generic_user_object
i
:
_init_args

	radial

angular
	indicator
;lin_transform"
_generic_user_object

<
_init_args
=instructions

>ls_max
?allowed_l_p
@	collector
Adownscale_embeddings
B
reducing_A"
_generic_user_object

C
_init_args
Dinstructions

Els_max
Fallowed_l_p
G	collector
Hdownscale_embeddings
I
reducing_A"
_generic_user_object

J
_init_args
Kinstructions

Lls_max
Mallowed_l_p
N	collector
Odownscale_embeddings
P
reducing_A"
_generic_user_object
k
Q
_init_args

	radial

angular
	indicator
Rcoupling_origin"
_generic_user_object

S
_init_args
Tinstructions

Uls_max
Vallowed_l_p
W	collector
Xdownscale_embeddings
Yreducing_YI"
_generic_user_object

Z
_init_args
[instructions

\ls_max
]allowed_l_p
^	collector
_downscale_embeddings
`
reducing_B"
_generic_user_object

a
_init_args
binstructions

cls_max
dallowed_l_p
e	collector
fdownscale_embeddings
g
reducing_B"
_generic_user_object
.
h
_init_args"
_generic_user_object
F
i
_init_args

target

jorigin"
_generic_user_object
b
k
_init_args

target

lorigin
mhidden_layers
nmlp"
_generic_user_object
:
o
_init_args

target"
_generic_user_object
+
	pshape"
trackable_dict_wrapper
+
	qshape"
trackable_dict_wrapper
+
	rshape"
trackable_dict_wrapper
+
	sshape"
trackable_dict_wrapper
+
	tshape"
trackable_dict_wrapper
+
	ushape"
trackable_dict_wrapper
+
	vshape"
trackable_dict_wrapper
+
	wshape"
trackable_dict_wrapper
+
	pshape"
trackable_dict_wrapper
+
	qshape"
trackable_dict_wrapper
+
	xshape"
trackable_dict_wrapper
+
	yshape"
trackable_dict_wrapper
+
	rshape"
trackable_dict_wrapper
+
	vshape"
trackable_dict_wrapper
+
	sshape"
trackable_dict_wrapper
+
	tshape"
trackable_dict_wrapper
+
	ushape"
trackable_dict_wrapper
+
	wshape"
trackable_dict_wrapper

z	capture_0
{	capture_1
|	capture_2
}	capture_5
~	capture_7
	capture_9

capture_11

capture_12

capture_13

capture_14

capture_15

capture_16

capture_18

capture_20

capture_22

capture_23

capture_24

capture_26

capture_27

capture_28

capture_30

capture_31

capture_32

capture_33

capture_34

capture_35

capture_36

capture_37

capture_38

capture_40

capture_41

capture_42

capture_44

capture_45

capture_46

capture_48

capture_49

capture_50
 
capture_51
Ё
capture_53
Ђ
capture_55
Ѓ
capture_56B
__inference_compute_143355atomic_mu_ibatch_tot_natbatch_tot_nat_realbond_vectorind_iind_jmu_imu_j"
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
 zz	capture_0z{	capture_1z|	capture_2z}	capture_5z~	capture_7z	capture_9z
capture_11z
capture_12z
capture_13z
capture_14z
capture_15z
capture_16z
capture_18z
capture_20z
capture_22z
capture_23z
capture_24z
capture_26z
capture_27z
capture_28z
capture_30z
capture_31z
capture_32z
capture_33z
capture_34z
capture_35z
capture_36z
capture_37z
capture_38z
capture_40z
capture_41z
capture_42z
capture_44z
capture_45z
capture_46z
capture_48z
capture_49z
capture_50z 
capture_51zЁ
capture_53zЂ
capture_55zЃ
capture_56
є
z	capture_0
{	capture_1
|	capture_2
}	capture_5
~	capture_7
	capture_9

capture_11

capture_12

capture_13

capture_14

capture_15

capture_16

capture_18

capture_20

capture_22

capture_23

capture_24

capture_26

capture_27

capture_28

capture_30

capture_31

capture_32

capture_33

capture_34

capture_35

capture_36

capture_37

capture_38

capture_40

capture_41

capture_42

capture_44

capture_45

capture_46

capture_48

capture_49

capture_50
 
capture_51
Ё
capture_53
Ђ
capture_55
Ѓ
capture_56Bѕ
$__inference_signature_wrapper_143490atomic_mu_ibatch_tot_natbatch_tot_nat_realbond_vectorind_iind_jmu_imu_j"ѕ
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
 zz	capture_0z{	capture_1z|	capture_2z}	capture_5z~	capture_7z	capture_9z
capture_11z
capture_12z
capture_13z
capture_14z
capture_15z
capture_16z
capture_18z
capture_20z
capture_22z
capture_23z
capture_24z
capture_26z
capture_27z
capture_28z
capture_30z
capture_31z
capture_32z
capture_33z
capture_34z
capture_35z
capture_36z
capture_37z
capture_38z
capture_40z
capture_41z
capture_42z
capture_44z
capture_45z
capture_46z
capture_48z
capture_49z
capture_50z 
capture_51zЁ
capture_53zЂ
capture_55zЃ
capture_56
 "
trackable_dict_wrapper
1
bond_length"
trackable_dict_wrapper
+
	bonds"
trackable_dict_wrapper
 "
trackable_dict_wrapper
'
Єrc"
_generic_user_object
+
	basis"
trackable_dict_wrapper
 "
trackable_list_wrapper
f
Ѕlayers_config
Іlayer0
Їlayer1
Јlayer2
Љlayer3"
_generic_user_object
*
vhat"
trackable_dict_wrapper
"
_generic_user_object
2
Њelement_map"
trackable_dict_wrapper
:2element_map_symbols
:2element_map_index
%:#2Z/ChemicalEmbedding
H
	indicator

angular

	radial"
trackable_dict_wrapper
&
Ћw"
_generic_user_object
R
Ќallowed_l_p
­instructions
Ўls_max"
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Џ0
А1"
trackable_list_wrapper
(
БA"
trackable_dict_wrapper
 "
trackable_dict_wrapper
&:$2I/reducing_A
R
Вallowed_l_p
Гinstructions
Дls_max"
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
Е0"
trackable_list_wrapper
(
ЖA"
trackable_dict_wrapper
 "
trackable_dict_wrapper
&:$2E/reducing_A
R
Зallowed_l_p
Иinstructions
Йls_max"
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
К0"
trackable_list_wrapper
(
ЛA"
trackable_dict_wrapper
 "
trackable_dict_wrapper
(:&2rho/reducing_A
Z

	radial
Мkeep_parity

angular
	indicator"
trackable_dict_wrapper
 "
trackable_list_wrapper
R
Нallowed_l_p
Оinstructions
Пls_max"
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
X
Р0
С1
Т2
У3
Ф4
Х5
Ц6"
trackable_list_wrapper
)
ЧYI"
trackable_dict_wrapper
 "
trackable_dict_wrapper
#:!2B/reducing_YI
R
Шallowed_l_p
Щinstructions
Ъls_max"
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
Ы0"
trackable_list_wrapper
(
ЬB"
trackable_dict_wrapper
 "
trackable_dict_wrapper
':%2E2/reducing_B
R
Эallowed_l_p
Юinstructions
Яls_max"
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
а0"
trackable_list_wrapper
(
бB"
trackable_dict_wrapper
 "
trackable_dict_wrapper
):'2rho2/reducing_B
 "
trackable_dict_wrapper
9
вorigin

target"
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
9
гorigin

target"
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
L
дlayers_config
еlayer0
жlayer1"
_generic_user_object
,

target"
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
"J

Const_41jtf.TrackableConstant
"J

Const_40jtf.TrackableConstant
"J

Const_39jtf.TrackableConstant
"J

Const_38jtf.TrackableConstant
"J

Const_37jtf.TrackableConstant
"J

Const_36jtf.TrackableConstant
"J

Const_35jtf.TrackableConstant
"J

Const_34jtf.TrackableConstant
"J

Const_33jtf.TrackableConstant
"J

Const_32jtf.TrackableConstant
"J

Const_31jtf.TrackableConstant
"J

Const_30jtf.TrackableConstant
"J

Const_29jtf.TrackableConstant
"J

Const_28jtf.TrackableConstant
"J

Const_27jtf.TrackableConstant
"J

Const_26jtf.TrackableConstant
"J

Const_25jtf.TrackableConstant
"J

Const_24jtf.TrackableConstant
"J

Const_23jtf.TrackableConstant
"J

Const_22jtf.TrackableConstant
"J

Const_21jtf.TrackableConstant
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
: 2cutoff
 "
trackable_list_wrapper
&
зw"
_generic_user_object
&
иw"
_generic_user_object
&
йw"
_generic_user_object
&
кw"
_generic_user_object
 "
trackable_dict_wrapper
::82(ChemIndTransf/DenseLayer_ChemIndTransf__
0
л0
м1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
н0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
о0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
п0
р1
с2
т3
у4
ф5
х6"
trackable_list_wrapper
X
ц0
ч1
ш2
щ3
ъ4
ы5
ь6"
trackable_list_wrapper
'
0"
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
trackable_dict_wrapper
(
э0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
ю0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
&
яw"
_generic_user_object
&
№w"
_generic_user_object
;:9@2)DenseLayer/DenseLayer_DenseLayer_no_decay
;:9@@2)DenseLayer/DenseLayer_DenseLayer_no_decay
;:9@@2)DenseLayer/DenseLayer_DenseLayer_no_decay
;:9@2)DenseLayer/DenseLayer_DenseLayer_no_decay
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
4:2 2"DenseLayer/DenseLayer_DenseLayer__
4:2 2"DenseLayer/DenseLayer_DenseLayer__
1b/
DenseLayer/Cast:0__inference_compute_143355
:b8
DenseLayer/einsum/Einsum:0__inference_compute_143355
3b1
DenseLayer/Cast_1:0__inference_compute_143355
<b:
DenseLayer/einsum_1/Einsum:0__inference_compute_143355
3b1
DenseLayer/Cast_2:0__inference_compute_143355
<b:
DenseLayer/einsum_2/Einsum:0__inference_compute_143355
3b1
DenseLayer/Cast_3:0__inference_compute_143355
<b:
DenseLayer/einsum_4/Einsum:0__inference_compute_143355х
__inference_compute_143355Цez{|Єз}и~йкЋ9BIPY`g яЁ№ЂЃяЂы
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
Њ "ъЊц
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
.
z_pair_f"
z_pair_fџџџџџџџџџю
#__inference_internal_grad_fn_143607Цёђ~Ђ{
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
tensor_2 ю
#__inference_internal_grad_fn_143634Цѓє~Ђ{
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
tensor_2 ю
#__inference_internal_grad_fn_143661Цѕі~Ђ{
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
tensor_2 ю
#__inference_internal_grad_fn_143688Цїј~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ 
(%
result_grads_1џџџџџџџџџ 

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ 

tensor_2 ш
$__inference_signature_wrapper_143490Пez{|Єз}и~йкЋ9BIPY`g яЁ№ЂЃшЂф
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
mu_jџџџџџџџџџ"ъЊц
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
.
z_pair_f"
z_pair_fџџџџџџџџџ