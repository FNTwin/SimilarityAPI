import numpy as np
import dscribe as ds #MAIN MODULE FOR SOAP FUNCTIONALITY
from ase.io import read
from ase import Atoms
from dscribe.descriptors import SOAP
import re


class Atom(object):
    def __init__(self, labels=None, pdb_line='', res_name='', atom_name='', atom_type='', element='', charge=0.0,
                 res_num=1, atom_num=1, x=0.0, y=0.0, z=0.0, kind='', zhi=None, chain_id='',
                 pytraj_atom=None, frame="none", traj="none", also_xyz=False):
        self.kind = kind
        self.atom_num = atom_num
        self.atom_name = atom_name
        self.altloc = ''
        self.res_name = res_name
        self.chainID = chain_id
        self.res_num = res_num
        self.x = x
        self.y = y
        self.z = z
        self.occupancy = 0.0
        self.tempfactor = 0.0
        self.element = element
        self.atom_type = atom_type
        self.charge = charge
        self.pdb_line = pdb_line
        self.np_coords, self.mycoords = None, None
        if labels: self.labels = labels
        else: self.labels = []
        if pdb_line: self.pdb_line_parser(in_line=pdb_line, zhi=zhi)
        if pytraj_atom: self.import_pytraj_atom(pytraj_atom=pytraj_atom, frame=frame,  traj=traj, also_xyz=also_xyz)
        self.distances = {}
        self.bonds = []
        self.neighbours = []

    def traslate(self, x=False, y=False, z=False):
        if x: self.x += x
        if y: self.y += y
        if z: self.z += z

    def scale(self, x=False, y=False, z=False):
        if x: self.x *= x
        if y: self.y *= y
        if z: self.z *= z

    def pdb_line_parser(self, in_line, reference=False, standard=True, return_labels=False, prev_at_idx=0, zhi=None):
        from numpy import array

        def add_with_label(items_list):
            for idx, value in enumerate(items_list):
                if self.labels[idx].lower() == 'kind': self.kind = value
                elif self.labels[idx].lower() == 'atom_num': self.atom_num = int(value)
                elif self.labels[idx].lower() == 'atom_name': self.atom_name = value
                elif self.labels[idx].lower() == 'altloc': self.altloc = value
                elif self.labels[idx].lower() == 'res_name': self.res_name = value
                elif self.labels[idx].lower() == 'chainid': self.chainID = value
                elif self.labels[idx].lower() == 'res_num': self.res_num = int(value)
                elif self.labels[idx].lower() == 'x': self.x = float(value)
                elif self.labels[idx].lower() == 'y': self.y = float(value)
                elif self.labels[idx].lower() == 'z': self.z = float(value)
                elif self.labels[idx].lower() == 'occupancy': self.occupancy = float(value)
                elif self.labels[idx].lower() == 'tempfactor': self.tempfactor = float(value)
                elif self.labels[idx].lower() == 'element': self.element = value
                elif self.labels[idx].lower() == 'charge': self.charge = float(value)
        # self.indx_limit = 99999
        if standard:
            self.kind = in_line[0:6].strip()
            try: self.atom_num = int(in_line[6:11].strip())
            except ValueError:
                if in_line[6:11] == "*****" and prev_at_idx:
                    self.atom_num = prev_at_idx + 1
                else:
                    self.atom_num = in_line[6:11].strip()
            self.atom_name = in_line[12:16].strip()
            self.altloc = in_line[16].strip()
            self.res_name = in_line[17:20].strip()
            if not reference:
                self.chainID = in_line[21].strip()
                self.res_num = int(in_line[22:28].strip())  # 22:26 standard... change also in the writer...
                self.x = float(in_line[30:38].strip())
                self.y = float(in_line[38:46].strip())
                self.z = float(in_line[46:54].strip())
                self.np_coords = array([self.x, self.y, self.z])
                try: self.occupancy = float(in_line[54:60].strip())
                except ValueError: pass
                try: self.tempfactor = float(in_line[60:66].strip())
                except ValueError: pass
                try: self.element = in_line[76:78].strip()
                except ValueError: pass
                try: self.charge = float(in_line[78:80].strip())
                except ValueError: pass
                if prev_at_idx: return self.atom_num
        else:
            in_line = in_line.replace('-', ' -')
            items = in_line.split()
            if len(self.labels) == len(items): add_with_label(items_list=items)
            else:
                self.labels = []
                print(">> Now I'll show the elements of the atom I've found.")
                print(">> For every element of the line identify the correct label from the following list:")
                print("> kind (ATOM/HETATM), atom_num, atom_name, altloc, res_name, chainID, res_num, x, y, z,")
                print("> occupancy, tempfactor, element, charge.")
                print(">> The line is: %s" % in_line)
                for item in items:
                    label = ''
                    while label == '': label = input("> '%s' has label: " % item)
                    self.labels.append(label)
                add_with_label(items_list=items)
            if return_labels: return self.labels
        if zhi:
            # requires a dict of type {zhi_at_name1: res_name1, ...}
            self.res_name_original = self.res_name
            try: self.res_name = zhi[self.atom_name]
            except KeyError:
                print(">> Error: zhi_atom_name %s not found in the input dictionary of residues" % self.atom_name)

    def import_pytraj_atom(self, pytraj_atom, traj="none", frame="none", also_xyz=False):
        from numpy import array
        self.atom_num = pytraj_atom.index + 1
        self.atom_name = pytraj_atom.name
        self.res_name = pytraj_atom.resname
        self.res_num = pytraj_atom.resid + 1
        self.atom_type = pytraj_atom.type
        self.charge = pytraj_atom.charge
        if frame != "none":
            if also_xyz:
                self.x = float(traj.xyz[frame][pytraj_atom.index][0])
                self.y = float(traj.xyz[frame][pytraj_atom.index][1])
                self.z = float(traj.xyz[frame][pytraj_atom.index][2])
            # print("adding coordinates to %s_%d" % (self.atom_name,self.atom_num))
            self.np_coords = array(traj.xyz[frame][pytraj_atom.index])

    def mol2_line_parser(self, in_line):
        # atom_id atom_name x y z atom_type [subst_id [subst_name [charge [status_bit]]]]
        self.mol2_items = in_line.strip('\n').split()
        self.atom_num, self.atom_name = int(self.mol2_items[0]), str(self.mol2_items[1])
        self.x, self.y, self.z = float(self.mol2_items[2]), float(self.mol2_items[3]), float(self.mol2_items[4])
        self.atom_type = str(self.mol2_items[5])
        if len(self.mol2_items) >= 7: self.subst_id = int(self.mol2_items[6])
        else: self.subst_id = 0
        if len(self.mol2_items) >= 8: self.subst_name = self.mol2_items[7]
        else: self.subst_name = ""
        if len(self.mol2_items) >= 9: self.charge = float(self.mol2_items[8])
        else: self.charge = 0.0
        if len(self.mol2_items) >= 10: self.status_bit = self.mol2_items[9]
        else: self.status_bit = ""

    def gro_line_parser(self, in_line, standard=True):
        from numpy import array
        if standard:
            self.kind = 'ATOM'
            self.res_num = int(in_line[0:5].strip())
            self.res_name = in_line[5:10].strip()
            self.atom_name = in_line[10:15].strip()
            self.atom_num = int(in_line[15:20].strip())
            self.x = float(in_line[20:28].strip())
            self.y = float(in_line[28:36].strip())
            self.z = float(in_line[36:44].strip())
            self.np_coords = array([self.x, self.y, self.z])

    def cof_line_parser(self, in_line, res_name='RES', res_num=0):
        from numpy import array
        self.cof_items = in_line.split()
        self.res_name, self.res_num = str(res_name), int(res_num)
        self.kind = 'HETATM'
        self.atom_num = int(self.cof_items[1])
        self.atom_name = self.cof_items[2]
        self.atom_type = self.cof_items[3]
        self.charge = float(self.cof_items[4])
        self.x = float(self.cof_items[5])
        self.y = float(self.cof_items[6])
        self.z = float(self.cof_items[7])
        self.np_coords = array([self.x, self.y, self.z])
        self.velocity_x = float(self.cof_items[8])
        self.velocity_y = float(self.cof_items[9])
        self.velocity_z = float(self.cof_items[10])
        self.velocity_const_x = float(self.cof_items[11])
        self.velocity_const_y = float(self.cof_items[12])
        self.velocity_const_z = float(self.cof_items[13])

    def xyz_line_parser(self, in_line, at_num=1, res_num=1, res_name="UNK"):
        items = in_line.split()
        self.kind = "HETATM"
        self.atom_num = at_num
        self.atom_name = items[0]
        self.res_name = res_name
        self.res_num = res_num
        self.x = float(items[1])
        self.y = float(items[2])
        self.z = float(items[3])

    def pdb_line_writer(self, out_file=None, discovery_studio=False):
        self.pdb_line = '{s.kind:<6s}'.format(s=self)
        try:
            if float(self.atom_num) > 99999 or self.atom_num == "*****":
                self.pdb_line += '***** {s.atom_name:>4s}'.format(s=self)
            else:
                if not discovery_studio:
                    self.pdb_line += '{s.atom_num:5d} {s.atom_name:<4s}'.format(s=self)
                else:
                    if len(self.atom_name) == 4 and self.atom_name.startswith("H"):
                        self.tmp_atom_name = self.atom_name[-1] + self.atom_name[:-1]
                        self.atom_name = self.tmp_atom_name
                        self.pdb_line += '{s.atom_num:5d} {s.atom_name:<4s}'.format(s=self)
                    else:
                        self.pdb_line += '{s.atom_num:5d}  {s.atom_name:<3s}'.format(s=self)

        except ValueError:
            self.pdb_line += '***** {s.atom_name:>4s}'.format(s=self)
        if self.altloc: self.pdb_line += '{s.altloc:1s}'.format(s=self)
        else: self.pdb_line += ' '
        self.pdb_line += '{s.res_name:3s} '.format(s=self)
        if self.chainID: self.pdb_line += '{s.chainID:1s}'.format(s=self)
        else: self.pdb_line += ' '
        coords = []
        for var in self.x, self.y, self.z:
            coord = '{:<7.4f}'.format(var).rstrip()
            if len(coord) >= 6 and coord[-1] == '0': coord = coord[:-1]
            if len(coord) >= 7 and "." in coord: coord = '{:<7.3f}'.format(float(coord))
            coords.append(coord)
        if self.res_num <= 9999:
            self.pdb_line += '{s.res_num:4d}    {c[0]:>8s}{c[1]:>8s}{c[2]:>8s}'.format(s=self, c=coords)
        elif self.res_num <= 99999:
            self.pdb_line += '{s.res_num:5d}   {c[0]:>8s}{c[1]:>8s}{c[2]:>8s}'.format(s=self, c=coords)
        elif self.res_num <= 999999:
            self.pdb_line += '{s.res_num:6d}  {c[0]:>8s}{c[1]:>8s}{c[2]:>8s}'.format(s=self, c=coords)
        elif self.res_num <= 9999999:
            self.pdb_line += '{s.res_num:7d} {c[0]:>8s}{c[1]:>8s}{c[2]:>8s}'.format(s=self, c=coords)
        if self.occupancy: self.pdb_line += '{s.occupancy:6.2f}'.format(s=self)
        else: self.pdb_line += '      '
        if self.tempfactor: self.pdb_line += '{s.tempfactor:6.2f}'.format(s=self)
        else: self.pdb_line += '      '
        if self.element: self.pdb_line += '{s.element:2s}'.format(s=self)
        else: self.pdb_line += '  '
        if self.charge: self.pdb_line += '{s.charge:8.5f}'.format(s=self)
        else: self.pdb_line += '        '
        self.pdb_line += '\n'
        if out_file: out_file.write(self.pdb_line)
        else: return self.pdb_line

    def gro_line_writer(self, out_file=None):
        self.gro_line = '{s.res_num:>5d}'.format(s=self)
        self.gro_line += '{s.res_name:<5s}'.format(s=self)
        self.gro_line += '{s.atom_name:>5s}'.format(s=self)
        self.gro_line += '{s.atom_num:>5d}'.format(s=self)
        self.gro_line += '{s.x:>8.3f}'.format(s=self)
        self.gro_line += '{s.y:>8.3f}'.format(s=self)
        self.gro_line += '{s.z:>8.3f}'.format(s=self)
        self.gro_line += '\n'
        if out_file: out_file.write(self.gro_line)
        else: return self.gro_line

    def cof_line_writer(self, out_file=None):
        self.cof_line = 'scbead\t{s.atom_num:d}'.format(s=self)
        self.cof_line += '\t{s.atom_name:s}'.format(s=self)
        self.cof_line += '\t{s.atom_type:s}'.format(s=self)
        self.cof_line += ('\t{s.charge:.4f}'.format(s=self)).rstrip('0').rstrip('.')
        self.cof_line += ('\t{s.x:.6f}'.format(s=self)).rstrip('0').rstrip('.')
        self.cof_line += ('\t{s.y:.6f}'.format(s=self)).rstrip('0').rstrip('.')
        self.cof_line += ('\t{s.z:.6f}'.format(s=self)).rstrip('0').rstrip('.')
        # if self.velocity_x or self.velocity_y or self.velocity_z:
        self.cof_line += ('\t{s.velocity_x:.6f}'.format(s=self)).rstrip('0').rstrip('.')
        self.cof_line += ('\t{s.velocity_y:.6f}'.    format(s=self)).rstrip('0').rstrip('.')
        self.cof_line += ('\t{s.velocity_z:.6f}'.format(s=self)).rstrip('0').rstrip('.')
        # if self.velocity_const_x or self.velocity_const_y or self.velocity_const_z:
        self.cof_line += ('\t{s.velocity_const_x:.6f}'.format(s=self)).rstrip('0').rstrip('.')
        self.cof_line += ('\t{s.velocity_const_y:.6f}'.format(s=self)).rstrip('0').rstrip('.')
        self.cof_line += ('\t{s.velocity_const_z:.6f}'.format(s=self)).rstrip('0').rstrip('.')
        self.cof_line += '\n'
        if out_file: out_file.write(self.cof_line)
        else: return self.cof_line

    def mol2_atom_line_writer(self, out_file=None):
        self.mol2_line = "%7d\t%5s\t%10.6f\t%10.6f\t%10.6f\t%5s" % (
            self.atom_num, self.atom_name, self.x, self.y, self.z, self.atom_type)
        if self.subst_id:
            self.mol2_line += "\t%7d" % self.subst_id
        if self.subst_name:
            self.mol2_line += "\t%8s" % self.subst_name
        if self.charge or self.charge == 0.0:
            self.mol2_line += "\t%10.4f" % self.charge
        if self.status_bit:
            self.mol2_line += "\t%s" % self.status_bit
        self.mol2_line += '\n'
        if out_file: out_file.write(self.mol2_line)
        else: return self.mol2_line

    def calc_distances(self, ref_coords, dist_name=None, sort=False, remove_self=True):
        from numpy import zeros, sqrt, float64, array
        self.mycoord = zeros(ref_coords.shape, dtype=float64)
        self.mycoord[:, :] = self.np_coords  # NB: np stands for NumPy... from the Atom class!
        if dist_name:
            self.distances[dist_name] = sqrt((self.mycoords[:, 0] - ref_coords[:, 0]) ** 2 +
                                             (self.mycoords[:, 1] - ref_coords[:, 1]) ** 2 +
                                             (self.mycoords[:, 2] - ref_coords[:, 2]) ** 2)
            if sort: self.distances[dist_name] = sorted(self.distances[dist_name])
        else:
            self.distances = sqrt((self.mycoord[:, 0] - ref_coords[:, 0]) ** 2 +
                                  (self.mycoord[:, 1] - ref_coords[:, 1]) ** 2 +
                                  (self.mycoord[:, 2] - ref_coords[:, 2]) ** 2)
            if sort: self.distances = sorted(self.distances)
            else:
                self.sorted_dist_idx = array([j[0] for j in sorted(enumerate(self.distances),
                                                                     key=lambda k: k[1])])  # (index_of_d, d)
                if remove_self: self.sorted_dist_idx = self.sorted_dist_idx[1:]

    def add_bond(self, at2, bond_type, bond_id=None, comment=None):
        self.tmp_bond = Bond(at1=self, at2=at2, bond_type=bond_type, bond_id=bond_id, comment=comment)
        self.bonds.append(self.tmp_bond)

    def add_neighbour(self, at2, neighbour_type, neighbour_id=None, comment=None):
        self.tmp_neighbour = Neighbour(at1=self, at2=at2, neighbour_type=neighbour_type, neighbour_id=neighbour_id,
                                       comment=comment)
        self.neighbours.append(self.tmp_neighbour)

    def add_spherical_coordinates(self, ISO=True, mathematics=False, from_zenith=True):
        from numpy import sqrt as np_sqrt, arccos as np_arccos, arctan2 as np_arctan2, arcsin as np_arcsin

        self.rho = np_sqrt((self.x ** 2) + (self.y ** 2) + (self.z ** 2))

        if from_zenith:
            # polar angle is the inclination from the z direction (from the zenith) [STANDARD]
            # arctan( sqrt(x^2+y^2) / z ) == arccot(z/sqrt(x^2+y^2))
            self.polar_angle = np_arccos(self.z / self.rho)  # output is in [0, pi]
        else:
            # polar angle is the elevation from the reference plane (XY), geographical, latitude
            self.polar_angle = np_arcsin(self.z / self.rho)  # output is in [-pi/2, pi/2]

        # np_arctan2(x1, x2) = arc tangent of x1/x2 choosing the quadrant correctly
        # longitude
        self.azimuthal_angle = np_arctan2(self.y, self.x)  # output is in [-pi, pi]

        # ISO, physics:     theta, inclination from the z direction, inclination from the zenith, polar angle
        #                   phi,   azimuthal angle from the z-axis (y-axis has phi = +pi/2)
        if ISO:
            self.theta = self.polar_angle
            self.phi = self.azimuthal_angle

        # other mathematics books: [same names, inverted roles...]
        #               theta, azimuthal angle from the z-axis (y-axis has phi=+pi/2)
        #               phi, inclination from the z direction, inclination from the zenith, polar angle
        elif mathematics:
            self.theta = self.azimuthal_angle
            self.phi = self.polar_angle

class Residue(object):
    def __init__(self):
        self.atoms = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.res_name = ''
        self.res_num = 1

    def add_atom(self, atom_to_add):
        self.atoms.append(atom_to_add)

#################################################################################################################
def parse_pdb_lines_list(pdb_lines_list, np_array=False, skip_res=""):
    atom_list, coords = [], []
    res_list = [Residue()]
    res_idx, line_idx, prev_at_idx = 0, 0, 1
    while line_idx < len(pdb_lines_list):
        line = pdb_lines_list[line_idx]
        if line.startswith('HETATM') or line.startswith('ATOM'):
            atom = Atom()
            prev_at_idx = atom.pdb_line_parser(in_line=line, prev_at_idx=prev_at_idx)
            if skip_res and atom.res_name == skip_res: pass
            elif np_array:
                atom_list.append(atom)
                coords.append([atom.x, atom.y, atom.z])
            else:
                if atom.res_num != res_idx:
                    res_idx += 1
                    res_list.append(Residue())
                    res_list[res_idx].add_atom(atom)
                    res_list[res_idx].res_name = res_list[res_idx].atoms[0].res_name
                    res_list[res_idx].res_num = res_list[res_idx].atoms[0].res_num
                else: res_list[res_idx].add_atom(atom)
            line_idx += 1
        else: line_idx += 1
    if np_array:
        from numpy import array, float64
        return atom_list, array(coords, dtype=float64)
    else:
        _ = res_list.pop(0)
        return res_list


def parse_xyz_lines_list(xyz_lines_list, at_idx=1, res_num=1, res_name='UNK', np_array=False):
    atom_list, coords = [], []
    for line in xyz_lines_list:
        if len(line.split()) == 4:
            atom = Atom()
            atom.xyz_line_parser(in_line=line, at_num=at_idx, res_num=res_num, res_name=res_name)
            at_idx += 1
            atom_list.append(atom)
            if np_array: coords.append([atom.x, atom.y, atom.z])
    if np_array:
        from numpy import array, float64
        return atom_list, array(coords, dtype=float64)
    else: return atom_list

################################################################################################################
async def pdb_parser(pdbtxt):
    allowed_residues=["WAT", "RAD"]
    atom_list=[]
    pdbtxt=pdbtxt.splitlines()
    for line in pdbtxt:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom = Atom()
            labels = atom.pdb_line_parser(in_line=line,  return_labels=True)
            if atom.res_name in allowed_residues:
                atom_list.append(atom)
    return atom_list

def create_universe(atom_list):
    xyz, label, name = [], [], []
    for atom in atom_list:
        xyz.append(np.array([atom.x,atom.y,atom.z]))
        label.append(atom.res_name)
        tmp=re.search("[a-zA-Z]+",atom.atom_name).group()
        if len(tmp)>=2:
            if tmp[1].isupper:
                tmp=tmp[0]
        name.append(tmp)
    return xyz,label,name

async def anal_traj(universe, soap, n_jobs=1):
    ####### PARSING XYZ AND INDEXS ##########
    xyz,label,name=universe
    xyz=np.array(xyz)
    label=np.array(label)
    name=np.array(name)
    wat_label= [label=="RAD"]
    PRB=label=="RAD"
    WAT=label=="WAT"

    traj_soap, pos_soap =[], []
    xyz_prb , xyz_wat = xyz[PRB], xyz[WAT] 

    ######## PROBE SOAP ###########
    prb_list=[]
    for pos,atom in zip(xyz_prb, name[PRB]):
        traj_soap.append(Atoms(symbols=atom, positions=[pos]))
        #prb_ase=Atoms(symbols=atom, positions=pos) #DOUBLE
        if atom=="N":
            pos_soap.append([pos])

    ######### WAT SOAP  ###########
    try:
        for pos,atom in zip(xyz_wat, name[WAT]):
            water_ase=Atoms(symbols="O", positions=[pos])
            traj_soap.append(water_ase)
    except:
        pass
    
    ########### CALCULATE SOAP ##############
    pos_list=[]
    for i in pos_soap:
        pos_list.extend(i)
    tot=traj_soap[0]
    for i,n in enumerate(traj_soap[1:],1):
        tot=tot+traj_soap[i]

    soap_tmp=soap.create(tot, positions=pos_list, n_jobs=n_jobs)
    return soap_tmp


def build_SOAP_descriptor_complete(data):
    chemical_species_complete=["S", "O", "N", "H", "C", "Au", "P", "F"]
    soap = SOAP(
        species=chemical_species_complete,
        periodic=False,
        rcut=9.0,
        nmax=8,
        lmax=8,
        sparse=True
    )
    #TODO: Implement this

async def build_SOAP_descriptor(data):

    chemical_species=["O", "N", "C", "H"]
    soap = SOAP(
        species=chemical_species,
        periodic=False,
        rcut=4.5,
        nmax=8,
        lmax=8,
        sparse=True
    ) 
    res=await anal_traj(data, soap, n_jobs=1)
    return res

