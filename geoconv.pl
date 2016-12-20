#!/usr/bin/perl

my $state = "";

my @mass_ary;

my @bond_kf_ary;
my @bond_r0_ary;
my @bond_ind_ary;

my @angl_kf_ary;
my @angl_r0_ary;
my @angl_ind_ary;

my %aidmap;
my $pdbfile = $ARGV[0];
open(IN, $pdbfile) || die "cannot open $pdbfile";

my @atoms;
my @anames;

my %res_st;
my %res_en;
#my $res_st=1000000;
#my $res_en=-1000000;

while (<IN>) {
    if (/^ATOM  /||/^HETATM/) {
# ATOM   1667  C   MET   304      17.355  18.183  28.553  1.00 17.31           C
#            " C   MET   304 "
	chomp;
	my $name = substr($_, 12, 15);
	my $id = int(substr($_, 6, 5))-1; 
	#print("name=<$name>, id=<$id>\n");
	$aidmap{$name} = $id;

	my $elem = substr($_, 76, 2);
	my $mass = 12;
	if ($elem eq " N") {
	    $mass = 14;
	}
	elsif ($elem eq " O") {
	    $mass = 16;
	}
	elsif ($elem eq " S") {
	    $mass = 32;
	}
	elsif ($elem eq " P") {
	    $mass = 30;
	}
	push(@atoms, $mass);

	my $aname = substr($name, 0, 4);
	my $rname = substr($name, 5, 3);
	my $cname = substr($name, 9, 1);
	my $sresid = substr($name, 10, 4);
	my $resid = int( $sresid );
	$anames[$id] = $aname;
	$rnames[$id] = $rname;
	$cnames[$id] = $cname;
	$rinds[$id] = $resid;

	print"chain $cname $res_st{$cname}\n";
	if ($res_st{$cname}) {
	    $res_st{$cname} = $resid if ($resid<$res_st{$cname});
	    $res_en{$cname} = $resid if ($resid>$res_en{$cname});
	}
	else {
	    $res_st{$cname} = $resid;
	    $res_en{$cname} = $resid;
	}

	$aidmap2{"$aname $cname$sresid "} = $id;
    }
}

foreach my $i (sort keys(%res_st)) {
    print("$i: resid start: $res_st{$i}\n");
    print("    resid end: $res_en{$i}\n");
}

my $geofile = $ARGV[1];
open(IN, $geofile) || die "cannot open $geofile";

#foreach my $i (keys %aidmap) {
#    print"$i==>$aidmap{$i}\n";
#}

sub toRadian($) {
    my $deg = shift;
    return $deg*(3.14159265358979323846)/180.0;
}

my @bonds;
my @angls;
my @dihes;
my @chirs;
while (<IN>) {
#bond pdb=" C   MET   304 "
#bond pdb=" N   ARG A  58 " segid="A   "
    if (/^bond pdb=\"([^\"]+)\"/) {
	my $id1 = $aidmap{$1};
	#print "bond <$1>, id1=$id1\n";
	$_=<IN>;
	die unless(/pdb=\"([^\"]+)\"/);
	my $id2 = $aidmap{$1};
	#print "     <$1>, id2=$id2\n";
	$_=<IN>;
	$_=<IN>;
#  ideal  model  delta    sigma   weight residual
#  1.524  1.802 -0.277 1.28e-02 6.10e+03 4.68e+02
	die unless(/([\-\.\d]+)\s+([\-\.\d]+)\s+([\-\.\d]+)\s+([e\-\+\.\d]+)\s+([e\-\+\.\d]+)\s+([e\-\+\.\d]+)/);
	my $r0 = $1;
	my $sig = $4;
	my $wgt = 1.0/($sig*$sig);
	push(@bonds, "$id1 $id2 $wgt $r0");
    }

    if (/^angle pdb=\"([^\"]+)\"/) {
	my $id1 = $aidmap{$1};
	#print "angle <$1>, id1=$id1\n";
	$_=<IN>;
	die unless(/pdb=\"([^\"]+)\"/);
	my $id2 = $aidmap{$1};
	#print "     <$1>, id2=$id2\n";

	$_=<IN>;
	die unless(/pdb=\"([^\"]+)\"/);
	my $id3 = $aidmap{$1};
	#print "     <$1>, id3=$id3\n";

	$_=<IN>;
	$_=<IN>;
#    ideal   model   delta    sigma   weight residual
#   120.79  141.90  -21.11 1.39e+00 5.18e-01 2.31e+02
	die unless(/([\-\.\d]+)\s+([\-\.\d]+)\s+([\-\.\d]+)\s+([e\-\+\.\d]+)\s+([e\-\+\.\d]+)\s+([e\-\+\.\d]+)/);
	my $r0 = toRadian($1);
	my $sig = toRadian($4);
	my $wgt = 1.0/($sig*$sig);
	#print "$4 sig=$sig wgt=$wgt\n";
	push(@angls, "$id1 $id2 $id3 $wgt $r0");
    }

    if (/^dihedral pdb=\"([^\"]+)\"/) {
	my $id1 = $aidmap{$1};
	#print "dihe <$1>, id1=$id1\n";
	$_=<IN>;
	die unless(/pdb=\"([^\"]+)\"/);
	my $id2 = $aidmap{$1};
	#print "     <$1>, id2=$id2\n";

	$_=<IN>;
	die unless(/pdb=\"([^\"]+)\"/);
	my $id3 = $aidmap{$1};
	#print "     <$1>, id3=$id3\n";

	$_=<IN>;
	die unless(/pdb=\"([^\"]+)\"/);
	my $id4 = $aidmap{$1};
	#print "     <$1>, id4=$id4\n";

	$_=<IN>;
	$_=<IN>;
#    ideal   model   delta sinusoidal    sigma   weight residual
#     0.00 -179.57  179.57     1      1.00e+01 1.00e-02 1.92e+02
	die unless(/([\-\.\d]+)\s+([\-\.\d]+)\s+([\-\.\d]+)\s+([\d]+)\s+([e\-\+\.\d]+)\s+([e\-\+\.\d]+)\s+([e\-\+\.\d]+)/);
	my $r0 = toRadian($1);
	my $peri = $4;
	my $sig = toRadian($5);
	my $wgt = 1.0/($sig*$sig);
	push(@dihes, "$id1 $id2 $id3 $id4 $wgt $r0 $peri");
    }


    if (/^chirality pdb=\"([^\"]+)\"/) {
	my $id1 = $aidmap{$1};
	#print "chir <$1>, id1=$id1\n";
	$_=<IN>;
	die unless(/pdb=\"([^\"]+)\"/);
	my $id2 = $aidmap{$1};
	#print "     <$1>, id2=$id2\n";

	$_=<IN>;
	die unless(/pdb=\"([^\"]+)\"/);
	my $id3 = $aidmap{$1};
	#print "     <$1>, id3=$id3\n";

	$_=<IN>;
	die unless(/pdb=\"([^\"]+)\"/);
	my $id4 = $aidmap{$1};
	#print "     <$1>, id4=$id4\n";

	$_=<IN>;
	$_=<IN>;

#  both_signs  ideal   model   delta    sigma   weight residual
#    False     -2.51   -1.94   -0.57 2.00e-01 2.50e+01 8.21e+00

	die unless(/(\w+)\s+([\-\.\d]+)\s+([\-\.\d]+)\s+([\-\.\d]+)\s+([e\-\+\.\d]+)\s+([e\-\+\.\d]+)\s+([e\-\+\.\d]+)/);
	my $sign;
	$sign = 1 if ($1 eq "True");
	$sign = 0 if ($1 eq "False");
	my $r0 = $2;
	my $sig = $5;
	my $wgt = 1.0/($sig*$sig);
	push(@chirs, "$id1 $id2 $id3 $id4 $wgt $r0 $sign");
    }

#plane pdb=" CA  TYR A1327 " segid="A   "    0.000 2.00e-02 2.50e+03   4.26e-05 1.81e-05
    if (/^plane pdb=\"([^\"]+)\"\s+([\-\.\d]+)\s+([e\-\+\.\d]+)\s+([e\-\+\.\d]+)/||
	/^plane pdb=\"([^\"]+)\"\s+segid=\"[^\"]+\"\s+([\-\.\d]+)\s+([e\-\+\.\d]+)\s+([e\-\+\.\d]+)/) {
	my @ids;
	my @sigs;

	my $id1 = $aidmap{$1};
	my $sig = $3;
	print "plan <$1>, id1=$id1 sig=$3\n";
	push(@ids, $id1);
	push(@sigs, $sig);
	while (<IN>) {
#      pdb=" C   TYR A1327 " segid="A   "   -0.000 2.00e-02 2.50e+03
	    if (/^\s+pdb=\"([^\"]+)\"\s+([\-\.\d]+)\s+([e\-\+\.\d]+)\s+([e\-\+\.\d]+)/||
		/^\s+pdb=\"([^\"]+)\"\s+segid=\"[^\"]+\"\s+([\-\.\d]+)\s+([e\-\+\.\d]+)\s+([e\-\+\.\d]+)/) {
		my $id1 = $aidmap{$1};
		my $sig = $3;
		print "   <$1>, id1=$id1 sig=$3\n";
		push(@ids, $id1);
		push(@sigs, $sig);
	    }
	    else {
		last;
	    }
	}
	my $nplan = int(@ids);
	my $s = "$nplan ";
	for (my $i=0; $i<$nplan; $i++) {
	    my $wgt = 1.0/($sigs[$i] * $sigs[$i]);
	    $s .= "$ids[$i] $wgt ";
	}

	# my $xxx = $aidmap{" N   PRO   331 "};
	# my $bOK = 0;
	# for (my $i=0; $i<$nplan; $i++) {
	# $bOK = 1 if ($xxx==$ids[$i]);
	# }
	# if ($bOK) {
	push(@plans, sprintf("%05d ", $ids[0]).$s);
	#i}
    }
}

my $parmfile = $ARGV[2];
open(OUT, ">$parmfile") || die "cannot open $parmfile";

my $natoms = int(@atoms);
print "natoms=$natoms\n";
print OUT "$natoms #NATOMS\n";
foreach my $i (@atoms) {
    print OUT "$i\n";
}

@bonds = sort {$a <=> $b} @bonds;
my $nbonds = int(@bonds);
print "nbonds=$nbonds\n";
print OUT "$nbonds #NBONDS\n";
foreach my $i (@bonds) {
    print OUT "$i\n";
}

@angls = sort {$a <=> $b} @angls;
my $nangls = int(@angls);
print "nangls = $nangls\n";
print OUT "$nangls #NANGLS\n";
foreach my $i (@angls) {
    print OUT "$i\n";
}

sub getID($$$) {
    my $cname = shift;
    my $resid = shift;
    my $aname = shift;
    my $sresid = sprintf("%4d", $resid);
    #print("$aname $cname$sresid \n");
    return $aidmap2{"$aname $cname$sresid "};
}

my @ramas;
my @omegs;

foreach my $ch (sort keys(%res_st)) {

for (my $i=$res_st{$ch}; $i<=$res_en{$ch}; ++$i) {
    my $i1 = $i;
    my $i2 = $i+1;
    my $i3 = $i+2;

    #print "check for $ch $i1-$i2-$i3\n";

    # omega
    my $omg_id1 = getID($ch, $i1, " CA ");
    my $omg_id2 = getID($ch, $i1, " C  ");
    my $omg_id3 = getID($ch, $i2, " N  ");
    my $omg_id4 = getID($ch, $i2, " CA ");
    my $omg_id5 = getID($ch, $i1, " O  ");
    if ($omg_id1&&
	$omg_id2&&
	$omg_id3&&
	$omg_id4&&
	$omg_id5) {
	#print "omega for $i1-$i2-$i3 found ($omg_id1 $omg_id2 $omg_id3 $omg_id4 $omg_id5)\n";

	my $r0 = toRadian(180.0);
	my $peri = 0;
	my $sig = toRadian(5.0);
	my $wgt = 1.0/($sig*$sig);
	#push(@omegs, "$omg_id1 $omg_id2 $omg_id3 $omg_id4 $wgt $r0 $peri");
	push(@plans, sprintf("%05d ", $omg_id1)."5 $omg_id1 2500 $omg_id2 2500 $omg_id3 2500 $omg_id4 2500 $omg_id5 2500");
    }

    # phi
    my $phi_id1 = getID($ch, $i1, " C  ");
    my $phi_id2 = getID($ch, $i2, " N  ");
    my $phi_id3 = getID($ch, $i2, " CA ");
    my $phi_id4 = getID($ch, $i2, " C  ");

    my $bphi = 0;
    if ($phi_id1&&
	$phi_id2&&
	$phi_id3&&
	$phi_id4) {
	$bphi = 1;
    }

    # psi
    my $psi_id1 = getID($ch, $i2, " N  ");
    my $psi_id2 = getID($ch, $i2, " CA ");
    my $psi_id3 = getID($ch, $i2, " C  ");
    my $psi_id4 = getID($ch, $i3, " N  ");

    my $bpsi = 0;
    if ($psi_id1&&
	$psi_id2&&
	$psi_id3&&
	$psi_id4) {
	$bpsi = 1;
    }

    if ($bphi && $bpsi) {
	#print "phi for $i1-$i2-$i3 found ($phi_id1 $phi_id2 $phi_id3 $phi_id4)\n";
	#print "psi for $i1-$i2-$i3 found ($psi_id1 $psi_id2 $psi_id3 $psi_id4)\n";

	push(@ramas, "$phi_id1 $phi_id2 $phi_id3 $phi_id4 $psi_id1 $psi_id2 $psi_id3 $psi_id4");
    }
}
}

# my $ndihes = int(@dihes);
# print OUT "$ndihes\n";
# foreach my $i (@dihes) {
#     print OUT "$i\n";
# }

my $nomegs = int(@omegs);
print "ndihes = $nomegs\n";
print OUT "$nomegs #dihes\n";
foreach my $i (@omegs) {
    print OUT "$i\n";
}

@chirs = sort {$a <=> $b} @chirs;
my $nchirs = int(@chirs);
print "nchirs = $nchirs\n";
print OUT "$nchirs # NCHIRS\n";
foreach my $i (@chirs) {
    print OUT "$i\n";
}

@plans = sort {$a <=> $b} @plans;
my $nplans = int(@plans);
print "NPLANS=$nplans\n";
print OUT "$nplans # NPLANS\n";
foreach my $i (@plans) {
    print OUT substr($i,6)."\n";
}

my $nramas = int(@ramas);
print "NRAMAS=$nramas\n";
print OUT "$nramas #NRAMAS\n";
foreach my $i (@ramas) {
    print OUT "$i\n";
}
