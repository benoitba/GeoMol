from ase.io import read
from scipy.spatial.distance import euclidean, cdist
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromPDBFile
from Bio.PDB.PDBParser import PDBParser
from collections import namedtuple
import numpy as np

Pattern = namedtuple('Pattern', ['smiles', 'index', 'max_size'])
CoarseAtom = namedtuple('CoarseAtom', ['index', 'position'])

class EGCMCalculator(object) :
    
    def __init__(self, 
                 patterns = (('C(O)=O', 30), 
                             ('O=CN', 65), 
                             ('NC(N)=N', 10), 
                             ('C1=CN=CN1', 10), 
                             ('C1=CNC2=C1C=CC=C2', 10),
                             ('C1=CC=C(O)C=C1', 20),
                             ('C1=CC=CC=C1', 20),
                             ('CN', 40),
                             ('CSC', 15),
                             ('CS', 15),
                             ('CO', 20)
                            )) :
        
        self.patterns = [Pattern(s, i + 1, ms) for i, (s, ms) in enumerate(patterns)]
        
    def get_egcm(self, ligand_mol, protein_path) :
    
        w = None
        try :
            #print(ligand_path)
            #ligand_config = read(ligand_path, format='mol')
            parser = PDBParser()
            protein = parser.get_structure('prot', protein_path)
            protein_rdmol = MolFromPDBFile(protein_path)

            if protein_rdmol is not None :

                pocket_residues = self.get_pocket_residues(protein, ligand_mol)

                atom_idxs = []
                for residue in pocket_residues :
                    atom_idxs = atom_idxs + [atom.get_serial_number() for atom in residue.get_atoms()]

                coarse_atoms = self.get_coarse_atoms(atom_idxs, protein_rdmol)

                coulomb_matrix = self.generate_coulomb_matrix(coarse_atoms)

                w, v = np.linalg.eig(coulomb_matrix)
                w.sort()
        except :
            print('Error on' + protein_path + ' ')

        return w
    
    def get_pocket_residues(self, protein, ligand_mol, cutoff=6.5) :
        pocket_residues = []
        ligand_positions = ligand_mol.GetConformer().GetPositions()
        for residue in protein.get_residues():
            for atom in residue :
                if cdist(atom.get_coord().reshape(1,-1), ligand_positions).min() < cutoff :
                    pocket_residues.append(residue)
                    break
        return pocket_residues
    
    def get_coarse_atoms(self, atom_idxs, protein_rdmol) : 
        protein_positions = protein_rdmol.GetConformer().GetPositions()
        coarse_atoms = []
        for pattern in self.patterns :
            mol_pattern = Chem.MolFromSmiles(pattern.smiles)
            matches = protein_rdmol.GetSubstructMatches(mol_pattern)
            current_coarses = []
            for match in matches :
                if all([(idx + 1) in atom_idxs for idx in match]) and len(current_coarses) < pattern.max_size :
                    match_positions = protein_positions[list(match)].mean(0)
                    current_coarses.append(CoarseAtom(pattern.index, match_positions))
            while len(current_coarses) < pattern.max_size :
                current_coarses.append(CoarseAtom(pattern.index, None))
            coarse_atoms = coarse_atoms + current_coarses
        return coarse_atoms
    
    def generate_coulomb_matrix(self, coarse_atoms) :
        size = len(coarse_atoms)
        coulomb_matrix = np.zeros((size, size))
        for i in range(size) :
            atom1 = coarse_atoms[i]
            for j in range(size) :
                atom2 = coarse_atoms[j]
                if i == j :
                    value = 0.5 * (atom1.index ** 2.4)
                elif atom1.position is not None and atom2.position is not None :
                    value = (atom1.index * atom2.index) / (euclidean(atom1.position, atom2.position))
                else :
                    value = 0
                coulomb_matrix[i][j] = value
        return coulomb_matrix
    
