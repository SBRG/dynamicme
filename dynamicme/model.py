#============================================================
# File model.py
#
# class ComplexDegradation
# class PeptideDegradation
# class ComplexDegradationData
# class PeptideDegradationData

# Model extensions
#
# Laurence Yang, SBRG, UCSD
#
# 05 Oct 2017:  migrated to separate file
#============================================================
from cobrame import StoichiometricData, ComplexData, SubreactionData
from cobrame import MEReaction
from cobra import DictList
from cobrame import mu

from six import iteritems, string_types
from collections import defaultdict

import warnings


class ProteaseData(ComplexData):
    """
    Complex degradation data
    """
    def __init__(self, id, model):
        ComplexData.__init__(self, id, model)
        self.enzyme = id
        self.target_data = []
        self.subreactions = defaultdict(int)
        self.lower_bound = 0.
        self.upper_bound = 1000.

    def create_subreaction_data(self, target_type):
        me = self._model
        if target_type == 'complex':
            data_id = 'Complex_degradation'
            stoichiometry = {'atp_c':-2, 'h2o_c':-2, 'adp_c':2, 'h_c':2}
            keff = 65.
        elif target_type in ['protein', 'peptide']:
            data_id = 'Protein_degradation'
            stoichiometry = {'h2o_c':-1}
            keff = 65.

        try:
            data = SubreactionData(data_id, me)
            data.stoichiometry = stoichiometry
            data.keff = keff
        except ValueError:
            data = me.subreaction_data.get_by_id(data_id)


    def create_degradation(self, target_type, verbose=True):
        """
        Create degradation reactions for all targets
        Inputs:
        target_type : complex or protein (peptide), str
        """
        me = self._model
        # Initialize subreactions
        self.create_subreaction_data(target_type)

        if target_type == 'complex':
            for target_data in self.target_data:
                degr_id = 'degradation_' + target_data.id + '_' + self.id
                if degr_id in self._model.reactions:
                    raise ValueError('reaction %s already in model' % degr_id)
                degr = ComplexDegradation(degr_id)
                degr.protease_data = self
                degr.complex_data = target_data
                degr.lower_bound = self.lower_bound
                degr.upper_bound = self.upper_bound
                self._model.add_reaction(degr)
                degr.update(verbose=verbose)
        elif target_type in ['protein', 'peptide']:
            for target_data in self.target_data:
                degr_id = 'degradation_' + target_data.protein + '_' + self.id
                if degr_id in self._model.reactions:
                    raise ValueError('reaction %s already in model' % degr_id)
                degr = PeptideDegradation(degr_id)
                degr.protease_data = self
                degr.translation_data = target_data
                degr.lower_bound = self.lower_bound
                degr.upper_bound = self.upper_bound
                self._model.add_reaction(degr)
                degr.update(verbose=verbose)
        else:
            raise ValueError('target_type must be complex or protein or peptide')


class ComplexDegradation(MEReaction):
    """
    Complex degradation reaction
    """
    def __init__(self, id):
        MEReaction.__init__(self, id)
        self._complex_data = None
        self.protease_data = None
        self.keff = 65.

    @property
    def complex_data(self):
        return self._complex_data

    @complex_data.setter
    def complex_data(self, process_data):
        if isinstance(process_data, string_types):
            process_data = self._model.complex_data.get_by_id(process_data)
        self._complex_data = process_data
        if not hasattr(process_data, 'complex_id'):
            raise TypeError('%s is not a ComplexData instance' %
                            process_data.id)
        if process_data is not None:
            process_data._parent_reactions.add(self.id)

    def update(self, verbose=True):
        me = self._model
        cplx_data = self.complex_data
        stoichiometry = defaultdict(float)
        for k,v in iteritems(cplx_data.stoichiometry):
            stoichiometry[k] = v
        stoichiometry[cplx_data.complex_id] = -1
        # Account for modifications
        for mod in cplx_data.modifications:
            for met,val in iteritems(me.modification_data.get_by_id(mod).stoichiometry):
                stoichiometry[met] = -val
        # Add subreaction
        self.protease_data.subreactions['Complex_degradation'] = 1
        stoichiometry = self.add_subreactions(self.protease_data.id, stoichiometry)
        # Catalyze using protease
        protease_data = self.protease_data
        stoichiometry[protease_data.id] = -mu / self.keff / 3600.
        object_stoichiometry = self.get_components_from_ids(
                stoichiometry, verbose=verbose)

        self.add_metabolites(object_stoichiometry, combine=False,
                add_to_container_model=False)


class PeptideDegradation(MEReaction):
    """
    Peptide degradation reaction
    """
    def __init__(self, id):
        MEReaction.__init__(self, id)
        self._translation_data = None
        self.protease_data = None
        self.keff = 65.

    @property
    def translation_data(self):
        return self._translation_data

    @translation_data.setter
    def translation_data(self, process_data):
        if isinstance(process_data, string_types):
            process_data = self._model.translation_data.get_by_id(process_data)
        self._translation_data = process_data
        if not hasattr(process_data, 'amino_acid_count'):
            raise TypeError('%s is not a TranslationData instance' %
                            process_data.id)
        if process_data is not None:
            process_data._parent_reactions.add(self.id)

    def update(self, verbose=True):
        self.clear_metabolites()
        stoichiometry = defaultdict(float)
        me = self._model
        translation_data = self.translation_data
        for aa, count in iteritems(translation_data.amino_acid_count):
            stoichiometry[aa] = count
        stoichiometry[translation_data.protein] = -1
        # Remove biomass of degraded metabolites
        stoichiometry[me._biomass.id] = -translation_data.mass
        # Add subreaction
        self.protease_data.subreactions['Protein_degradation'] = \
                len(translation_data.amino_acid_sequence)-1
        stoichiometry = self.add_subreactions(self.protease_data.id, stoichiometry)

        # Catalyze using protease
        protease_data = self.protease_data
        stoichiometry[protease_data.id] = -mu / self.keff / 3600.
        object_stoichiometry = self.get_components_from_ids(
                stoichiometry, verbose=verbose)

        self.add_metabolites(object_stoichiometry, combine=False,
                add_to_container_model=False)
