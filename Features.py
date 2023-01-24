import abc
import numpy as np
import os
import json
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
import pdb
import itertools
from tqdm import tqdm

class DualMol:
    """
    Class to deal with a molecule representation in both rdkit implementation and pybel implementation.
    needs to be instantiated with an xyz file
    """

    def __init__(self, xyzfilepath):
        self.original_inchi = None
        if "InChI=" in xyzfilepath:
            self.pymol = pybel.readstring('inchi', xyzfilepath)
            self.pymol.localopt()
            self.original_inchi = xyzfilepath
        else:
            xyzfiletext = open(xyzfilepath, 'r').read()
            self.pymol = pybel.readstring('xyz', xyzfiletext)
        
        self.molecular_formula = self.pymol.formula
        
        # pdbfiletext = self.pymol.write("pdb")
        # fileout = open("tempfile.pdb", 'w+').write(pdbfiletext)
        # temp_molecule = Chem.MolFromPDBFile('tempfile.pdb')
        # os.remove("tempfile.pdb")
        pdbfiletext = self.pymol.write("smi")
        
        temp_molecule = Chem.MolFromSmiles(pdbfiletext)
        if temp_molecule is None:
            try:
                temp_molecule = Chem.MolFromSmiles(self.pymol.write("smi"))
            except:
                temp_molecule = self.pymol.removeh()
                temp_molecule = Chem.MolFromSmiles(temp_molecule.pymol.write("smi"))
        try:
            temp_molecule = Chem.rdmolops.AddHs(temp_molecule)
        except:
            print("couldn't add Hydrogens, implicit hydrogens might be missing")
            
        self.rdmol = temp_molecule
        if temp_molecule is None:
            self.rdmol = None
            self.pymol = None

class Featurizer:
    def __init__(self):
        self.fundict = {
                        "topo": self.calc_topological_vec,
                        "morgan": self.calc_mfp,
                        "hom-lum": self.get_egap_homo_lumo,
                        "homo": self.get_homo,
                        "lumo": self.get_lumo,
                        "gibbs": self.get_gibbs,
                        "entropy": self.get_entropy,
                        'min_lumo_reactants': self.get_lumo,
                        'max_lumo_reactants': self.get_lumo,
                        'max_h-l_reactants': self.get_egap_homo_lumo,
                        'min_homo_reactants': self.get_homo,
                        'max_homo_reactants': self.get_homo,
                        'min_lumo_products': self.get_lumo,
                        'max_lumo_products': self.get_lumo,
                        'max_h-l_products': self.get_egap_homo_lumo,
                        'min_homo_products': self.get_homo,
                        'max_homo_products': self.get_homo,
                        
                        'max_lumo_alpha_reactants': self.get_lumo_alpha,
                        'min_lumo_alpha_reactants': self.get_lumo_alpha,
                        
                        'min_homo_alpha_reactants': self.get_homo_alpha,
                        'max_homo_alpha_reactants': self.get_homo_alpha,

                        'min_lumo_beta_products': self.get_lumo_beta,
                        'max_lumo_beta_products': self.get_lumo_beta,
                        
                        'min_homo_beta_products': self.get_homo_beta,
                        'max_homo_beta_products': self.get_homo_beta,
                        
                        }

        self.units_convert_si = {'Ea [J/mol]': 1,
                                 'Ea [cal/mol]': 4.186798,
                                 'Ea [kJ/mol]': 1000,
                                 'Ea [kcal/mol]': 4186.798}
        self.main_energy_dict = None
        self.energydict = None

        self.homolumoenergydict = None
        self.main_homolumoenergydict = None

        self.homodict = None
        self.main_homodict = None

        self.lumodict = None
        self.main_lumodict = None
        self.configfile = json.load(open("config.json","r"))
        self.basedir = os.getcwd()
        self.trainingdir = self.configfile["training_set_directory"]
        self.reactionsdb = self.configfile["searches_directory"]
        self.gibbsdict = json.loads(open(self.configfile["species_db"],"r").read())
        self.gibbsdict_dict = self.gibbsdict

        self.moleculedir = self.configfile["species_db"]
        self.dft_results = self.configfile["dft_results"]
        self.molecdict = self.configfile["molecular_dict"]
        self.inchikey_dict = {}
        self.default_families = [
            'H_Abstraction',
        ]

    def update_molecular_dict(self, out=False):
        count_good = 0
        resdict = {}
        currentdir = os.getcwd()
        os.chdir(self.dft_results)
        for i in tqdm(os.listdir()):
            os.chdir(self.dft_results)
            if not (os.path.isdir(i)):
                continue
            os.chdir(i)
            if "relaxed_" + i + ".xyz" in os.listdir():
                try:
                    molecule = DualMol("relaxed_" + i + ".xyz")
                except:
                    continue
                if self.fundict["gibbs"](i, molecule) is not None:
                    count_good += 1
                    molecular_descriptors ={}
                    for keyword_descriptor in self.fundict.keys():
                        if'gibbs' in keyword_descriptor or 'bond' in keyword_descriptor or\
                                'homo' in keyword_descriptor or 'lumo' in keyword_descriptor or\
                                "hom-lum" in keyword_descriptor or 'h-l' in keyword_descriptor:
                            continue
                    try:
                        molecular_descriptors.update({"molecformula": molecule.pymol.formula,
                                                      "morgan":self.fundict["morgan"](molecule),
                                                      "topo":self.fundict["topo"](molecule),
                                                      "gibbs": self.fundict["gibbs"](i, molecule),
                                                      "enthalpy": self.fundict["gibbs"](i,molecule),
                                                      "entropy": self.fundict["entropy"](i,molecule),
                                                      "lumo": self.fundict["lumo"](i),
                                                      "homo": self.fundict["homo"](i),
                                                      'hom-lum': self.fundict["hom-lum"](i),
                                                      'min_lumo_reactants': self.fundict["min_lumo_reactants"](i),
                                                      'max_lumo_reactants': self.fundict["max_lumo_reactants"](i),
                                                      'max_h-l_reactants': self.fundict['max_h-l_reactants'](i),
                                                      'min_homo_reactants': self.fundict["min_homo_reactants"](i),
                                                      'max_homo_reactants': self.fundict["max_homo_reactants"](i),
                                                      'min_lumo_products': self.fundict["min_lumo_reactants"](i),
                                                      'max_lumo_products': self.fundict["max_lumo_reactants"](i),
                                                      'max_h-l_products': self.fundict['max_h-l_reactants'](i),
                                                      'min_homo_products': self.fundict["min_homo_reactants"](i),
                                                      'max_homo_products': self.fundict["max_homo_reactants"](i), 
                                                      
                                                      # 'max_lumo_alpha_reactants':self.fundict["max_lumo_alpha_reactants"](i),
                                                      # 'min_lumo_alpha_reactants':self.fundict["min_lumo_alpha_reactants"](i),
                            
                                                      # 'min_homo_alpha_reactants':self.fundict["min_homo_alpha_reactants"](i),
                                                      # 'max_homo_alpha_reactants':self.fundict["max_homo_alpha_reactants"](i),

                                                      # 'min_lumo_beta_products':self.fundict["min_lumo_beta_products"](i),
                                                      # 'max_lumo_beta_products':self.fundict["max_lumo_beta_products"](i),
                            
                                                      # 'min_homo_beta_products':self.fundict["min_homo_beta_products"](i),
                                                      # 'max_homo_beta_products':self.fundict["max_homo_beta_products"](i),
                                                      
                                                    })
                    except:
                        continue
                    resdict[i] = molecular_descriptors
                else:
                    print(i,'is None')
            else:
                print(i,'is none ')
            os.chdir(self.dft_results)
        if out:
            os.chdir(self.basedir)
            try:
                with open("molec_descriptor.dict", 'w')as fileout:
                    json.dump(resdict, fileout, indent=2)
            except:
                pdb.set_trace()

        print('full info molecules', str(count_good))
        os.chdir(currentdir)
        return resdict

    @staticmethod
    def calc_mfp(mol, rad=5, bitvect=True):
        """
        calculates morgan fingerprint for given milecule
        :param mol: molecule for which the morgan fingerprint is to be calculated
        :param rad: maximum bond length explored in morgan fingerprint
        :return:
        """
        if mol is None:
            return None
        if bitvect:
            res = np.array(AllChem.GetMorganFingerprintAsBitVect(mol.rdmol, rad)).tolist()
            return res
        res = AllChem.GetMorganFingerprint(mol.rdmol, rad)
        return res

    @staticmethod
    def calc_topological_vec(mol, k=None):
        """
        Description
        ------------------------------------------------------------
        takes in a pybel molecule and gives back the eigenvalues of
        the adjacency matrix elevated to the powers given in list
        k.

        Input
        ------------------------------------------------------------
        mol : pybel molecule
        k   : [] list of powers to which the eigenvalues of the
            adjacency matrix should be taken. Default taken as
            [1, 2, 3, 4, 5, 'inf'].

        ------------------------------------------------------------
        Output : list of sum eigenvalues elevated to different powers
        """
        if mol is None:
            return None

        if k is None:
            k = [1, 2, 3, 4, 5, 'inf']
            # k = [3]

        res = np.zeros((len(k), 1))
        m = mol.rdmol
        try:
            a = Chem.rdmolops.GetAdjacencyMatrix(m, useBO=True)
        except:
            return None
        eigs = np.linalg.eig(a)[0]
        for i in range(len(k)):
            mk = 0.0
            for eig in eigs:
                if k[i] != 'inf':
                    mk += abs(eig) ** k[i]
                elif (k[i] == 'inf') or (k[i] == np.Infinity):
                    if abs(eig) > 1:
                        mk += 1
            res[i] = mk
        return res.flatten().tolist()

    def get_egap_homo_lumo(self, inchikey):
        """
        Returns the homo-lumo energy gap for a molecule
        :param mol: molecule for which the homo-lumo energy gap is to be retrieved.
        :return: None if molecule not found / float energy if found
        """
        return self.get_lumo(inchikey) - self.get_homo(inchikey)

    @staticmethod
    def adder(prods, react):
        """
        operation products - reactants producing a reaction fingerprint
        :param prods: list of reactant fingerprints
        :param react: list of reactant fingerprints
        :return: reaction fingerprint
        """
        for i in [*prods, *react]:
            if i is None:
                return None
        res = np.zeros(np.array(prods[0]).shape)
        for i in prods:
            res = res + i
        for i in react:
            res = res - i
        return res

    @staticmethod
    def fp_stdizer(fplist):
        res = []
        if not (type(fplist) is list):
            fplist = [fplist]
        for i in range(len(fplist)):
            if fplist[i] is None:
                res.append(None)
                continue
            keys = list(fplist[i].keys())
            for key in keys:
                if type((fplist[i][key])) in [float, np.float64]:
                    res.append(fplist[i][key])
                else:
                    res = res + list(fplist[i][key])
        return np.array(res)

    def listreader(self, strlist):
        if type(strlist) is not str:
            return np.array(strlist)
        strlist = strlist.replace("[", "").replace("]", "").replace("\\n", "").replace(",", "").split(" ")
        res = np.array([float(i) for i in strlist if i != ""])
        return res

    def check_molecule_name(self, inchikey):
        res = inchikey
        currentdir = os.getcwd()
        os.chdir(self.dft_results)
        os.chdir(inchikey)
        molec = DualMol("relaxed_" + inchikey + ".xyz")
        inchi_new = molec.pymol.write("inchikey").replace("\n", "")
        if inchikey != inchi_new:
            res = inchi_new
        os.chdir(currentdir)
        return res

    def strict_to_list(self, numpy_array):
        if type(numpy_array) is not np.ndarray:
            raise TypeError("Only numpy arrays can be converted into lists with this method")
        res = numpy_array.tolist()
        if type(res) is not list:
            res = [res]
        return res

    def trainingsetgenerator(self, out=False, features=None):

        important_indices = np.array([0, 6, 7, 14, 15, 19, 45, 50, 62, 69, 75, 88, 89,
                                      91, 92, 94, 98, 99, 104, 106, 108, 111, 114, 115, 117, 120,
                                      129, 138, 140, 144, 146, 167, 168, 172, 185, 191, 195, 198, 203,
                                      214, 220, 222, 230, 253, 257, 260, 262, 265, 271, 274, 279, 283,
                                      289, 290, 293, 301, 305, 312, 316, 317, 318, 319, 328, 332, 336,
                                      340, 341, 347, 350, 353, 368, 370, 373, 388, 407, 414, 415, 416,
                                      423, 429, 433, 435, 437, 440, 441, 447, 450, 452, 456, 462, 463,
                                      468, 472, 480, 484, 486, 489, 492, 497, 498])

        if features is None:
            features = ["topo", "morgan"]
        featurizer = Featurizer()
        name = "".join(i + "_" for i in features[0:7])
        xvalmname = name + ".trainfeatures"
        yvalname = name + ".trainvalues"

        rxnlist, rxnstr, fplist, ealist, index_list = [], [], [], [], []
        database_file = "reactions.json"
        reaction_info = json.loads(open(os.path.join(featurizer.reactionsdb,
                                                     database_file), 'r').read())
        molecule_feature_info = json.loads(open(self.molecdict, 'r').read())
        molecule_feature_info_list = list(molecule_feature_info.keys())

        good_reaction_indices = []
        index_explained = {}

        for index_of_reaction,reaction in tqdm(enumerate(reaction_info)):
            feature_vector = []
            for k in features:
                maxmin_feature = False
                if 'min' in k:
                    maxmin_feature = True
                    maxmin_operator = min
                elif 'max' in k:
                    maxmin_feature = True
                    maxmin_operator = max

                products = list(reaction["ProdInChI"])
                reactants = list(reaction["ReacInChI"])

                all_species = reactants + products
                badreaction = False
                molecs_outside_info_list = [i for i in all_species if i not in molecule_feature_info_list]
                prod_vals = []
                reac_vals = []

                if molecs_outside_info_list:
                    continue
                for specie in reactants:
                    if 'morgan' in k:
                        specie_val = np.array(self.listreader(molecule_feature_info[specie][k]))
                        reac_vals.append(specie_val[important_indices])
                    else:
                        reac_vals.append(self.listreader(molecule_feature_info[specie][k]))
                for specie in products:
                    if 'morgan' in k:
                        specie_val = np.array(self.listreader(molecule_feature_info[specie][k]))
                        prod_vals.append(specie_val[important_indices])
                    else:
                        prod_vals.append(self.listreader(molecule_feature_info[specie][k]))
                if maxmin_feature:
                    if 'reactants' in k:
                        val = maxmin_operator(reac_vals)
                    elif 'products':
                        val = maxmin_operator(prod_vals)
                else:
                    val = np.sum(np.array(prod_vals),axis=0)-np.sum(np.array(reac_vals),axis=0)

                if type(val) is list or type(val) is np.ndarray:
                    feature_vector = feature_vector + self.strict_to_list(val)
                else:
                    feature_vector.append(val)
                try:
                    index_explained[k] = len(val)
                except:
                    index_explained[k] = 1
            ea = None
            for reaction_data in list(reaction.keys()):
                if "Ea" in reaction_data:
                    ea = float(reaction[reaction_data]) * float(featurizer.units_convert_si[reaction_data])

            if ea is not None and len(feature_vector) > 0: #ea is not none and feature_vector is not empty
                good_reaction_indices.append(reaction["rxn_index"])
                ealist.append(ea)
                fplist.append(feature_vector)
                index_list.append(index_of_reaction)

        if out:
            fplist_out = np.array(fplist)
            ealist_out = np.array(ealist)
            fplist_out.dump(os.path.join(featurizer.trainingdir, xvalmname))
            ealist_out.dump(os.path.join(featurizer.trainingdir, yvalname))
            open(os.path.join(self.trainingdir,'reaction_indices_out.json'), 'w').write(json.dumps(good_reaction_indices, indent=2))
            
            total_features = np.sum(list(index_explained.values()))
            cumulative_features = {}
            index_current = 0
            final_feature_indices = {}
            for i in index_explained.keys():
                cumulative_features[i] = index_current+index_explained[i]-1
                index_current += index_explained[i]
            print(cumulative_features)
            for i in range(total_features):
                for j in cumulative_features.keys():
                    if cumulative_features[j] >= i and i > (cumulative_features[j] - index_explained[j]):
                        final_feature_indices[i] = j
            open(os.path.join(self.trainingdir,'features_explained.json'), 'w').write(json.dumps(final_feature_indices, indent=2))

        return [np.array(fplist), np.array(ealist), np.array(index_list)]
    
    def get_homo(self, inchikey):
        try:
            res = self.gibbsdict_dict[inchikey]['HOMO']
            return res
        except:
            return None
    
    def get_lumo(self, inchikey):
        try:
            res = self.gibbsdict_dict[inchikey]['LUMO']
            return res
        except:
            return None
    
    def get_lumo_alpha(self, inchikey):
        try:
            res = self.gibbsdict_dict[inchikey]["alpha_lumo"]
            if res == 0:
                res = res = self.get_lumo(inchikey)
            return res
        except:
            return None
    
    def get_lumo_beta(self, inchikey):
        try:
            res = self.gibbsdict_dict[inchikey]["beta_lumo"]
            return res
        except:
            return None
    
    def get_homo_alpha(self, inchikey):
        try:
            res = self.gibbsdict_dict[inchikey]["alpha_homo"]
            if res == 0:
                res = self.get_homo(inchikey)
            return res
        except:
            return None
    
    def get_homo_beta(self, inchikey):
        try:
            res = self.gibbsdict_dict[inchikey]["beta_homo"]
            return res
        except:
            return None

    def get_gibbs(self, inchikey, molecule, T=300):
        """
        Returns the Gibb's free energyfor a molecule
        :param mol: molecule for which the homo-lumo energy gap is to be retrieved.
        :return: None if molecule not found / float energy if found
        """
        res = None
        currentdir = os.getcwd()
        os.chdir(self.dft_results)
        os.chdir(inchikey)
        try:
            res = self.gibbsdict_dict[inchikey]['G0']
            return res
        except:
            return None
         
    def get_entropy(self, inchikey, molecule, T=300):
        """
        Returns the Gibb's free energyfor a molecule
        :param mol: molecule for which the homo-lumo energy gap is to be retrieved.
        :return: None if molecule not found / float energy if found
        """
        res = None
        currentdir = os.getcwd()
        os.chdir(self.dft_results)
        os.chdir(inchikey)
        try:
            res = self.gibbsdict_dict[inchikey]['dG300']
            return res
        except:
            return None

    @staticmethod
    def inchi_main(inchi1, sections=3):
        section_no_1 = len(re.findall("/", inchi1)) + 1
        sections1 = inchi1.split('/')
        sections = min(sections, section_no_1)
        res = ''
        for i in range(sections):
            res = res + sections1[i] + "/"
        # print(res)
        res = res[:-1]
        return res