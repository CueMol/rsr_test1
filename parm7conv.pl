#!/usr/bin/perl

my $state = "";

my @mass_ary;

my @bond_kf_ary;
my @bond_r0_ary;
my @bond_ind_ary;

my @angl_kf_ary;
my @angl_r0_ary;
my @angl_ind_ary;

while (<>) {
    if (/^%FLAG\s+(\w+)/) {
	my $type = $1;
	print "Flag: ".$type."\n";
	$state = $type;
    }
    elsif (/^%FORMAT/) {
	print;
    }
    elsif ($state eq "MASS") {
	my @ln = split;
	foreach my $i (@ln) {
	    #print "$i\n";
	    push(@mass_ary, $i);
	}
    }

    # BOND
    elsif ($state eq "BOND_FORCE_CONSTANT") {
	my @ln = split;
	foreach my $i (@ln) {
	    #print "$i\n";
	    push(@bond_kf_ary, $i);
	}
    }
    elsif ($state eq "BOND_EQUIL_VALUE") {
	my @ln = split;
	foreach my $i (@ln) {
	    #print "$i\n";
	    push(@bond_r0_ary, $i);
	}
    }
    elsif ($state eq "BONDS_INC_HYDROGEN" ||
	   $state eq "BONDS_WITHOUT_HYDROGEN") {
	my @ln = split;
	foreach my $i (@ln) {
	    #print "$i\n";
	    push(@bond_ind_ary, $i);
	}
    }

    # ANGLE
    elsif ($state eq "ANGLE_FORCE_CONSTANT") {
	my @ln = split;
	foreach my $i (@ln) {
	    #print "$i\n";
	    push(@angl_kf_ary, $i);
	}
    }
    elsif ($state eq "ANGLE_EQUIL_VALUE") {
	my @ln = split;
	foreach my $i (@ln) {
	    #print "$i\n";
	    push(@angl_r0_ary, $i);
	}
    }

    elsif ($state eq "ANGLES_INC_HYDROGEN" ||
	   $state eq "ANGLES_WITHOUT_HYDROGEN") {
	my @ln = split;
	foreach my $i (@ln) {
	    #print "$i\n";
	    push(@angl_ind_ary, $i);
	}
    }

    # DIHEDRAL
    elsif ($state eq "DIHEDRAL_FORCE_CONSTANT") {
	my @ln = split;
	foreach my $i (@ln) {
	    #print "$i\n";
	    push(@dihe_kf_ary, $i);
	}
    }

    elsif ($state eq "DIHEDRAL_PERIODICITY") {
	my @ln = split;
	foreach my $i (@ln) {
	    #print "$i\n";
	    push(@dihe_per_ary, $i);
	}
    }

    elsif ($state eq "DIHEDRAL_PHASE") {
	my @ln = split;
	foreach my $i (@ln) {
	    #print "$i\n";
	    push(@dihe_pha_ary, $i);
	}
    }

    elsif ($state eq "DIHEDRALS_INC_HYDROGEN" ||
	   $state eq "DIHEDRALS_WITHOUT_HYDROGEN") {
	my @ln = split;
	foreach my $i (@ln) {
	    #print "$i\n";
	    push(@dihe_ind_ary, $i);
	}
    }

}

my $natom = int(@mass_ary);
print("Natom=$natom\n"); 
print STDERR ("$natom\n"); 
for (my $i=0; $i<$natom; ++$i) {
    my $m = $mass_ary[$i];
    #print("atom $i : mass=$m\n");
    print STDERR ("$m\n");
}
#print("Mass: ".join(" ", @mass_ary)."\n");
#print("\n");
#print("Bond kf: ".join(" ", @bond_kf_ary)."\n");
#print("\n");
#print("Bond r0: ".join(" ", @bond_r0_ary)."\n");
#print("\n");

my $nbond = int(@bond_ind_ary)/3;
print("Nbond=$nbond\n"); 
print STDERR ("$nbond\n"); 
for (my $i=0; $i<$nbond; ++$i) {
    my $ii = $bond_ind_ary[$i*3+0]/3;
    my $jj = $bond_ind_ary[$i*3+1]/3;
    my $ti = $bond_ind_ary[$i*3+2]-1;
    my $kf = $bond_kf_ary[$ti];
    my $r0 = $bond_r0_ary[$ti];
    print("bond $ii <--> $jj : $ti, kf=$kf, r0=$r0\n");
    print STDERR ("$ii $jj $kf $r0\n");
}


my $nangl = int(@angl_ind_ary)/4;
print("Nangl=$nangl\n"); 
print STDERR ("$nangl\n"); 
for (my $i=0; $i<$nangl; ++$i) {
    my $ii = $angl_ind_ary[$i*4+0]/3;
    my $jj = $angl_ind_ary[$i*4+1]/3;
    my $kk = $angl_ind_ary[$i*4+2]/3;
    my $ti = $angl_ind_ary[$i*4+3]-1;
    my $kf = $angl_kf_ary[$ti];
    my $r0 = $angl_r0_ary[$ti];
    print("angl $ii <--> $jj <--> $kk : $ti, kf=$kf, r0=$r0\n");
    print STDERR ("$ii $jj $kk $kf $r0\n");
}


my $ndihe = int(@dihe_ind_ary)/5;
print("Ndihe=$ndihe\n"); 
print STDERR ("$ndihe\n"); 
for (my $i=0; $i<$ndihe; ++$i) {
    my $ii = $dihe_ind_ary[$i*5+0]/3;
    my $jj = $dihe_ind_ary[$i*5+1]/3;
    my $kk = $dihe_ind_ary[$i*5+2];
    my $ll = $dihe_ind_ary[$i*5+3];
    my $ti = $dihe_ind_ary[$i*5+4]-1;
    my $kf = $dihe_kf_ary[$ti];
    my $per = $dihe_per_ary[$ti];
    my $pha = $dihe_pha_ary[$ti];
    print("dihe $ii <--> $jj <--> $kk <--> $ll: $ti, kf=$kf, per=$per, pha=$pha\n");
    print STDERR ("$ii $jj $kk $ll $kf $per $pha\n");
}
