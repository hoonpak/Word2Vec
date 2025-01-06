import numpy as np

class HS_CBOW:
    def __init__(self, vocab_size, hidden_size, id2vocab):
        # if init:
        #     stddev = np.sqrt(2.0 / (vocab_size + hidden_size))
        #     self.W_in = np.random.randn(vocab_size, hidden_size).astype("f") * stddev
        #     self.W_out = np.random.randn(vocab_size, hidden_size).astype("f") * stddev
        # else:
        #     self.W_in = np.random.randn(vocab_size, hidden_size).astype("f")
        #     self.W_out = np.random.randn(vocab_size, hidden_size).astype("f")
        self.W_in = np.random.uniform(-0.5, 0.5, size = (vocab_size, hidden_size)).astype("f")
        self.W_in /= hidden_size
        self.W_out = np.zeros_like(self.W_in).astype("f")
        
        self.id2vocab = id2vocab
        self.hidden_size = hidden_size
        
        self.cache = None
        
        self.word_vec = self.W_in
        
    def update(self, context, target, lr):
        """
        Args:
            context: window*2
            target: 1
        """
        context = np.array(context)
        real_idx = context[context != -1]
        
        h1 = self.W_in[real_idx].sum(axis=0)
        h1 /= len(real_idx)
        
        vocab_points = self.id2vocab[target].point
        vocab_codes = self.id2vocab[target].code
        vocab_codelen = self.id2vocab[target].codelen
        
        loss = 0
        loss_len = 0
        dh1 = np.zeros_like(self.W_in[0])
        for i in range(vocab_codelen):
            ph = self.W_out[vocab_points[i]]
            f = np.matmul(ph,h1.T)
            if f >= 6 or f <= -6: continue
            f = 1/(1+np.exp(-f))
            loss += -(1-vocab_codes[i])*np.log(f + 1e-7) - (vocab_codes[i])*np.log(1 - f + 1e-7)
            g = (1 - vocab_codes[i] - f)*lr
            dh1 += g*ph
            self.W_out[vocab_points[i]] += g*h1
            loss_len += 1
        
        np.add.at(self.W_in,real_idx,dh1)
        if loss_len == 0: loss_len = 1
        loss /= loss_len
        return loss

class HS_Skipgram:
    def __init__(self, vocab_size, hidden_size, id2vocab):
        self.W_in = np.random.uniform(-0.5, 0.5, size = (vocab_size, hidden_size)).astype("f")
        self.W_in /= hidden_size
        self.W_out = np.zeros_like(self.W_in).astype("f")
        
        self.id2vocab = id2vocab
        self.hidden_size = hidden_size
        
        self.cache = None
        
        self.word_vec = self.W_in
        
    def update(self, context, target, lr):
        """
        Args:
            context: window*2
            target: 1
        """
        vocab_points = self.id2vocab[target].point
        vocab_codes = self.id2vocab[target].code
        vocab_codelen = self.id2vocab[target].codelen
        
        total_loss = 0
        for c in context:
            if c == -1: continue
            dh1 = np.zeros_like(self.W_in[0])
            h1 = self.W_in[c]
            
            loss = 0
            loss_len = 0
            for i in range(vocab_codelen):
                ph = self.W_out[vocab_points[i]]
                f = np.matmul(ph,h1.T)
                if f >= 6 or f <= -6: continue
                f = 1/(1+np.exp(-f))
                loss += -(1-vocab_codes[i])*np.log(f + 1e-7) - (vocab_codes[i])*np.log(1 - f + 1e-7)
                g = (1 - vocab_codes[i] - f)*lr
                dh1 += g*ph
                self.W_out[vocab_points[i]] += g*h1
                loss_len += 1
                
            self.W_in[c] += dh1
            if loss_len == 0: loss_len = 1
            total_loss += (loss/loss_len)
        return total_loss
    
class NO_LOSS_HS_CBOW:
    def __init__(self, vocab_size, hidden_size, id2vocab):
        # if init:
        #     stddev = np.sqrt(2.0 / (vocab_size + hidden_size))
        #     self.W_in = np.random.randn(vocab_size, hidden_size).astype("f") * stddev
        #     self.W_out = np.random.randn(vocab_size, hidden_size).astype("f") * stddev
        # else:
        #     self.W_in = np.random.randn(vocab_size, hidden_size).astype("f")
        #     self.W_out = np.random.randn(vocab_size, hidden_size).astype("f")
        self.W_in = np.random.uniform(-0.5, 0.5, size = (vocab_size, hidden_size)).astype("f")
        self.W_in /= hidden_size
        self.W_out = np.zeros_like(self.W_in).astype("f")
        
        self.id2vocab = id2vocab
        self.hidden_size = hidden_size
        
        self.cache = None
        
        self.word_vec = self.W_in
        
    def update(self, context, target, lr):
        """
        Args:
            context: window*2
            target: 1
        """
        context = np.array(context)
        real_idx = context[context != -1]
        
        h1 = self.W_in[real_idx].sum(axis=0)
        h1 /= len(real_idx)
        
        vocab_points = self.id2vocab[target].point
        vocab_codes = self.id2vocab[target].code
        vocab_codelen = self.id2vocab[target].codelen
        
        dh1 = np.zeros_like(self.W_in[0])
        for i in range(vocab_codelen):
            ph = self.W_out[vocab_points[i]]
            f = np.matmul(ph,h1.T)
            if f >= 6 or f <= -6: continue
            f = 1/(1+np.exp(-f))
            g = (1 - vocab_codes[i] - f)*lr
            dh1 += g*ph
            self.W_out[vocab_points[i]] += g*h1
        
        np.add.at(self.W_in,real_idx,dh1)
        return None

class NO_LOSS_HS_Skipgram:
    def __init__(self, vocab_size, hidden_size, id2vocab):
        self.W_in = np.random.uniform(-0.5, 0.5, size = (vocab_size, hidden_size)).astype("f")
        self.W_in /= hidden_size
        self.W_out = np.zeros_like(self.W_in).astype("f")
        
        self.id2vocab = id2vocab
        self.hidden_size = hidden_size
        
        self.cache = None
        
        self.word_vec = self.W_in
        
    def update(self, context, target, lr):
        """
        Args:
            context: window*2
            target: 1
        """
        vocab_points = self.id2vocab[target].point
        vocab_codes = self.id2vocab[target].code
        vocab_codelen = self.id2vocab[target].codelen
        
        for c in context:
            if c == -1: continue
            
            dh1 = np.zeros_like(self.W_in[0])
            h1 = self.W_in[c]
            
            for i in range(vocab_codelen):
                ph = self.W_out[vocab_points[i]]
                f = np.matmul(ph,h1.T)
                if f >= 6 or f <= -6: continue
                f = 1/(1+np.exp(-f))
                g = (1 - vocab_codes[i] - f)*lr
                dh1 += g*ph
                self.W_out[vocab_points[i]] += g*h1
                
            self.W_in[c] += dh1
        return None
    