from abc import ABC, abstractmethod
import k2
import torch

from diac_btc.config import WILDCARD_TOKEN, NO_DIAC_TOKEN, UNK_DIAC_TOKEN

class DiacritizationModel(ABC):
    '''
    Base class for diacritization models. implements
    - load_model
    - get_logits
    - diacritize ctc
    - diacritize wfst
    '''

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def get_logits(self):
        pass

    @property
    def token2id(self):
        pass

    @property
    def constrained_wildcard_ids(self):
        pass

    @property
    def unconstrained_wildcard_ids(self):
        pass

    @property
    def word_delimiter_token(self):
        pass
    @property
    def ctc_topo(self):
        return k2.arc_sort(k2.ctc_topo(len(self.token2id)))

    def build_pattern_fsa(self, pattern, constrained=True):
        '''
        Build pattern FSA.
        Args:
            pattern: pattern to build
            wildcard_ids: wildcard ids
            token2id: token to id mapping
        Returns:
            pattern FSA
        '''
        wildcard_ids = self.constrained_wildcard_ids if constrained else self.unconstrained_wildcard_ids
        token2id = self.token2id
        arcs = []
        state = 0
        for i, ch in enumerate(pattern):
            if ch == WILDCARD_TOKEN:
                for wid in wildcard_ids:
                    arcs.append(f"{state} {state+1} {wid} {wid} 0.0")
            else:
                if ch not in token2id:
                    continue
                tid = token2id[ch]
                arcs.append(f"{state} {state+1} {tid} {tid} 0.0")
            state += 1
        arcs.append(f"{state} 0.0")
        txt = "\n".join(arcs)
        fsa = k2.Fsa.from_str(txt, acceptor=False, openfst=True)
        return k2.arc_sort(fsa), txt
    
    def decode_wfst(self, logits, pattern, constrained=True, search_beam=20.0, output_beam=8.0, min_active_states=30, max_active_states=10000):
        '''
        Decode WFST.
        Args:
            logits: logits from the model
            pattern: pattern to decode
            wildcard_ids: wildcard ids
            constrained: whether to use constrained wildcard ids
            token2id: token to id mapping
        Returns:
            decoded text
        '''
        fsa, _ = self.build_pattern_fsa(pattern, constrained=constrained)

        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        dense = k2.DenseFsaVec(log_probs.unsqueeze(0), torch.tensor([[0, 0, T]], dtype=torch.int32))
        ctc_topo = k2.arc_sort(k2.ctc_topo(V))
        decoding_graph = k2.arc_sort(k2.compose(ctc_topo, fsa))
        lattice = k2.intersect_dense_pruned(
            decoding_graph, dense,
            search_beam=search_beam, 
            output_beam=output_beam,
            min_active_states=min_active_states, 
            max_active_states=max_active_states
        )
        best_path = k2.shortest_path(lattice, use_double_scores=False)
        aux = k2.get_aux_labels(best_path)[0]
        hyp_ids = [x for x in aux if x >= 0]

        if len(hyp_ids) == 0:
            return ""

        result = "".join(self.id2token[i] for i in hyp_ids)
        
        return result
    
    def decode_ctc(self, logits, search_beam=20.0, output_beam=8.0, min_active_states=30, max_active_states=10000):
        '''
        Decode CTC.
        Args:
            logits: logits from the model
            pattern: pattern to decode
            constrained: whether to use constrained wildcard ids
        Returns:
            decoded text
        '''
        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        dense = k2.DenseFsaVec(log_probs.unsqueeze(0), torch.tensor([[0, 0, T]], dtype=torch.int32))
        ctc_topo = k2.arc_sort(k2.ctc_topo(V-1))
        lattice = k2.intersect_dense_pruned(
            ctc_topo, dense,
            search_beam=search_beam, 
            output_beam=output_beam,
            min_active_states=min_active_states, 
            max_active_states=max_active_states
        )
        best_path = k2.shortest_path(lattice, use_double_scores=False)
        aux = k2.get_aux_labels(best_path)[0]
        hyp_ids = [x for x in aux if x >= 0]
        result = "".join(self.id2token[i] for i in hyp_ids)

        # replace word delimiter with space
        if self.word_delimiter_token:
            result = result.replace(self.word_delimiter_token, " ")
            
        return result

    def decode_ctc_greedy(self, logits):
        '''
        Decode CTC greedy.
        Args:
            logits: logits from the model
        Returns:
            decoded text
        '''
        log_probs = torch.log_softmax(logits, dim=-1)
        greedy_ids = log_probs.argmax(dim=-1)
        result = "".join(self.id2token[i] for i in greedy_ids)
        if self.word_delimiter_token:
            result = result.replace(self.word_delimiter_token, " ")
        return result

    @abstractmethod
    def diacritize(self, text, audio_path, constrained=True, method="wfst"):
        raise NotImplementedError("Subclasses must implement this method")