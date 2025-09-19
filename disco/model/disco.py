import tensorflow as tf
import sys
# sys.path.insert(0, '../utils/')
sys.path.append('../utils')
from utils.utils import softmax, init_weights, calc_catNLL, calc_gaussian_KL, calc_gaussian_KL_simple, D_KL, mse, calc_mode, drop_out,l1_l2_norm_calculation
import numpy as np
import copy

class DISCO:
    """
        The proposed DisCo model. See paper for details.

        @author DisCo Authors
    """
    def __init__(self, xi_dim, yi_dim, ya_dim, y_dim, a_dim, lat_i_dim=20, lat_a_dim=30,
                 lat_dim=10, act_fx="softsign", init_type="gaussian", name="var",
                 i_dim=-1, lat_fusion_type="sum", drop_p=0.0, gamma_i=1.0, gamma_a=1.0,l1_norm=0.0,l2_norm=0.0, meta_dim=768):
        self.name = name
        self.seed = 69
        self.gamma_i = gamma_i #1.0
        self.gamma_a = gamma_a #1.0
        self.lat_fusion_type = lat_fusion_type
        self.i_dim = i_dim
        self.a_dim = a_dim
        self.y_dim = y_dim
        self.xi_dim = xi_dim
        self.yi_dim = yi_dim
        self.ya_dim = ya_dim
        self.lat_dim = lat_dim
        self.lat_i_dim = lat_i_dim
        self.lat_a_dim = lat_a_dim
        self.drop_p = drop_p #0.5
        self.l1_norm = l1_norm
        self.l2_norm = l2_norm
        self.meta_dim = meta_dim  # Metadata embedding dimension

        # Validate dropout rate
        if drop_p < 0.0 or drop_p > 1.0:
            print(f"Warning: Invalid dropout rate {drop_p}. Clamping to [0.0, 1.0]")
            drop_p = max(0.0, min(1.0, drop_p))
        self.drop_p = drop_p #0.5
        print(f"Model initialized with dropout rate: {self.drop_p}")

        if self.lat_fusion_type != "concat":
            self.lat_a_dim = self.lat_i_dim
            print(" > Setting lat_a.dim equal to lat_i.dim (dim = {0})".format(self.lat_i_dim))

        self.act_fx = act_fx
        self.fx = None
        if act_fx == "tanh":
            self.fx = tf.nn.tanh
        elif act_fx == "sigmoid":
            self.fx = tf.nn.sigmoid
        elif act_fx == "relu":
            self.fx = tf.nn.relu
        elif act_fx == "relu6":
            self.fx = tf.nn.relu6
        elif act_fx == "leaky_relu":
            self.fx = tf.nn.leaky_relu
        elif act_fx == "elu":
            self.fx = tf.nn.elu
        elif act_fx == "swish":
            self.fx = tf.nn.swish
        else:
            self.fx = tf.identity
        self.fx_y = softmax
        self.fx_yi = softmax
        self.fx_ya = softmax

        stddev = 0.05 # 0.025
        self.theta_y = []

        self.Wi = init_weights(init_type, [self.xi_dim,self.lat_i_dim], self.seed, stddev=stddev)
        self.Wi = tf.Variable(self.Wi, name="Wi", dtype=tf.float32)
        self.theta_y.append(self.Wi)

        # Updated to handle metadata embeddings instead of just annotator IDs
        # a_dim now represents the metadata embedding dimension
        self.Wa = init_weights(init_type, [self.a_dim,self.lat_a_dim], self.seed, stddev=stddev)
        self.Wa = tf.Variable(self.Wa, name="Wa", dtype=tf.float32)
        self.theta_y.append(self.Wa)

        bot_dim = self.lat_i_dim
        if self.lat_fusion_type == "concat":
            bot_dim = self.lat_i_dim + self.lat_a_dim

        self.Wp = init_weights(init_type, [bot_dim,self.lat_dim], self.seed, stddev=stddev)
        self.Wp = tf.Variable(self.Wp, name="Wp", dtype=tf.float32)
        self.theta_y.append(self.Wp)

        self.We = None
        #if collapse_We is False:
        self.We = init_weights(init_type, [self.lat_dim,self.lat_dim], self.seed, stddev=stddev)
        self.We = tf.Variable(self.We, name="We", dtype=tf.float32)
        self.theta_y.append(self.We)

        self.Wy = init_weights(init_type, [self.lat_dim,self.y_dim], self.seed, stddev=stddev)
        self.Wy = tf.Variable(self.Wy, name="Wy", dtype=tf.float32)
        self.theta_y.append(self.Wy)

        self.Wyi = init_weights(init_type, [self.lat_dim,self.yi_dim], self.seed, stddev=stddev)
        self.Wyi = tf.Variable(self.Wyi, name="Wyi", dtype=tf.float32)
        self.theta_y.append(self.Wyi)

        self.Wya = init_weights(init_type, [self.lat_dim,self.ya_dim], self.seed, stddev=stddev)
        self.Wya = tf.Variable(self.Wya, name="Wya", dtype=tf.float32)
        self.theta_y.append(self.Wya)

        self.z_i = tf.Variable(tf.zeros([1,self.lat_dim]), name="z_i", dtype=tf.float32)
        self.z_a = tf.Variable(tf.zeros([1,self.lat_dim]), name="z_a", dtype=tf.float32)

        self.eta_v = 0.002
        self.moment_v = 0.9
        adam_eps = 1e-7 #1e-8  1e-6
        self.y_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.eta_v,beta1=0.9, beta2=0.999, epsilon=adam_eps)

    def set_opt(self, opt_type, eta_v, moment_v=0.9):
        adam_eps = 1e-7
        self.eta_v = eta_v
        self.moment_v = moment_v
        if opt_type == "adam":
            self.y_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.eta_v,beta1=0.9, beta2=0.999, epsilon=adam_eps)
        elif opt_type == "rmsprop":
            self.y_opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.eta_v,decay=0.9, momentum=self.moment_v, epsilon=1e-6)
        else:
            self.y_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.eta_v)

    def calc_loss(self, y, yi, ya, y_prob, yi_prob, ya_prob, use_evaluation_loss=False, alpha=0.5, objective_type="combined", step=0):
        """
        Calculate loss with different objective types for specialized training.
        
        Args:
            y, yi, ya: True distributions
            y_prob, yi_prob, ya_prob: Predicted distributions
            use_evaluation_loss: Whether to use evaluation metrics as loss
            alpha: Weight for soft label vs perspectivist evaluation (if objective_type="combined")
            objective_type: "soft", "perspectivist", "combined", "alternating", or "original"
            step: Current training step (used for alternating objective)
        """
        if use_evaluation_loss:
            from utils.utils import multi_objective_loss, alternating_loss
            
            if objective_type == "alternating":
                # Use alternating loss
                Ly = alternating_loss(y, y_prob, step, num_classes=self.y_dim)
                Lyi = alternating_loss(yi, yi_prob, step, num_classes=self.yi_dim) * self.gamma_i
                Lya = alternating_loss(ya, ya_prob, step, num_classes=self.ya_dim) * self.gamma_a
            else:
                # Use specialized objective loss
                Ly = multi_objective_loss(y, y_prob, objective_type, num_classes=self.y_dim, alpha=alpha)
                Lyi = multi_objective_loss(yi, yi_prob, objective_type, num_classes=self.yi_dim, alpha=alpha) * self.gamma_i
                Lya = multi_objective_loss(ya, ya_prob, objective_type, num_classes=self.ya_dim, alpha=alpha) * self.gamma_a
            
        else:
            # Original loss calculation
            Ly = calc_catNLL(target=y,prob=y_prob,keep_batch=True)
            Ly = tf.reduce_mean(Ly)
            Lyi = D_KL(yi, yi_prob) * self.gamma_i
            Lya = D_KL(ya, ya_prob) * self.gamma_a
        
        # Regularization terms (same for all approaches)
        l1 = 0.0
        l2 = 0.0
        mini_bath_size = y.shape[0] * 1.0
        if self.l1_norm > 0:
            l1 = l1_l2_norm_calculation(self.theta_y,1,mini_bath_size) * self.l1_norm 
        if self.l2_norm > 0:
            l2 = l1_l2_norm_calculation(self.theta_y,2,mini_bath_size) * self.l2_norm 
        
        L_t = Ly + Lyi + Lya + l1 + l2
        return L_t, Ly, Lyi, Lya

    def calc_evaluation_loss(self, y, yi, ya, y_prob, yi_prob, ya_prob, alpha=0.5):
        """
        Calculate loss using evaluation metrics (soft label + perspectivist).
        
        Args:
            y, yi, ya: True distributions
            y_prob, yi_prob, ya_prob: Predicted distributions
            alpha: Weight for soft label vs perspectivist evaluation
        
        Returns:
            Total loss and individual components
        """
        from utils.utils import average_WS_loss, mean_absolute_distance_loss
        
        # Calculate individual evaluation losses
        Ly_soft = average_WS_loss(y, y_prob)
        Ly_persp = mean_absolute_distance_loss(y, y_prob, num_classes=self.y_dim)
        Ly = alpha * Ly_soft + (1 - alpha) * Ly_persp
        
        Lyi_soft = average_WS_loss(yi, yi_prob)
        Lyi_persp = mean_absolute_distance_loss(yi, yi_prob, num_classes=self.yi_dim)
        Lyi = (alpha * Lyi_soft + (1 - alpha) * Lyi_persp) * self.gamma_i
        
        Lya_soft = average_WS_loss(ya, ya_prob)
        Lya_persp = mean_absolute_distance_loss(ya, ya_prob, num_classes=self.ya_dim)
        Lya = (alpha * Lya_soft + (1 - alpha) * Lya_persp) * self.gamma_a
        
        # Regularization terms
        l1 = 0.0
        l2 = 0.0
        mini_bath_size = y.shape[0] * 1.0
        if self.l1_norm > 0:
            l1 = l1_l2_norm_calculation(self.theta_y,1,mini_bath_size) * self.l1_norm 
        if self.l2_norm > 0:
            l2 = l1_l2_norm_calculation(self.theta_y,2,mini_bath_size) * self.l2_norm 
        
        L_t = Ly + Lyi + Lya + l1 + l2
        
        # Return detailed breakdown
        return L_t, Ly, Lyi, Lya, {
            'Ly_soft': Ly_soft, 'Ly_persp': Ly_persp,
            'Lyi_soft': Lyi_soft, 'Lyi_persp': Lyi_persp,
            'Lya_soft': Lya_soft, 'Lya_persp': Lya_persp,
            'l1': l1, 'l2': l2
        }

    def encode_i(self, xi):
        """
            Calculates projection/embedding of item feature vector x_i
        """
        # Ensure input tensor has the same dtype as the weight matrix
        xi = tf.cast(xi, self.Wi.dtype)
        z_enc = tf.matmul(xi, self.Wi)
        return z_enc

    def encode_a(self, a):
        """
            Calculates projection/embedding of annotator metadata
            a: annotator metadata embeddings [batch_size, meta_dim]
        """
        # Ensure input tensor has the same dtype as the weight matrix
        a = tf.cast(a, self.Wa.dtype)
        # Direct matrix multiplication since a contains metadata embeddings
        z_enc = tf.matmul(a, self.Wa)
        return z_enc

    def encode(self, xi, a):
        z = None
        if self.lat_fusion_type == "concat":
            z = self.fx(tf.concat([self.encode_i(xi), self.encode_a(a)],axis=1))
        else:
            z = self.fx(self.encode_i(xi) + self.encode_a(a))
        z = self.transform(z)
        return z

    def transform(self,z):
        z_e = z
        z_p = self.fx(tf.matmul(z, self.Wp))
        if self.drop_p > 0.0:
            # Debug: print dropout rate
            if self.drop_p >= 1.0:
                print(f"WARNING: Dropout rate is {self.drop_p}, which will cause division by zero!")
            z_p, _ = drop_out(z_p, rate=self.drop_p)
        z_e = self.fx(tf.matmul(z_p, self.We) + z_p)
        if self.drop_p > 0.0:
            # Debug: print dropout rate
            if self.drop_p >= 1.0:
                print(f"WARNING: Dropout rate is {self.drop_p}, which will cause division by zero!")
            z_e, _ = drop_out(z_e, rate=self.drop_p)
        return z_e

    def decode_yi(self, z):
        y_logits = tf.matmul(z, self.Wyi)
        y_dec = self.fx_yi(y_logits)
        return y_dec, y_logits

    def decode_ya(self, z):
        y_logits = tf.matmul(z, self.Wya)
        y_dec = self.fx_ya(y_logits)
        return y_dec, y_logits

    def decode_y(self, z):
        y_logits = tf.matmul(z, self.Wy)
        y_dec = self.fx_y(y_logits)
        return y_dec, y_logits

    def update(self, xi, a, yi, ya, y, update_radius=-1., use_evaluation_loss=False, alpha=0.5, objective_type="combined", step=0):
        """
            Updates model parameters given data batch (i, a, yi, ya, y)
        """
        batch_size = yi.shape[0]

        # run the model under gradient-tape's awareness
        with tf.GradientTape(persistent=True) as tape:
            z = self.encode(xi, a)
            yi_prob, yi_logits = self.decode_yi(z)
            ya_prob, ya_logits = self.decode_ya(z)
            y_prob, y_logits = self.decode_y(z)

            if use_evaluation_loss:
                # Use evaluation metrics as loss
                L_t, Ly, Lyi, Lya = self.calc_loss(y, yi, ya, y_prob, yi_prob, ya_prob, use_evaluation_loss=True, alpha=alpha, objective_type=objective_type, step=step)
            else:
                # Use original loss calculation
                Ly = calc_catNLL(target=y,prob=y_prob,keep_batch=True)
                Ly = tf.reduce_mean(Ly)
                Lyi = D_KL(yi, yi_prob) * self.gamma_i
                Lya = D_KL(ya, ya_prob) * self.gamma_a
                
                l1 = 0.0
                l2 = 0.0
                mini_bath_size = y.shape[0] * 1.0
                if self.l1_norm  > 0:
                    l1 = l1_l2_norm_calculation(self.theta_y,1,mini_bath_size) * self.l1_norm
                if self.l2_norm  > 0:
                    l2 = l1_l2_norm_calculation(self.theta_y,2,mini_bath_size) * self.l2_norm
                L_t = Ly + Lyi + Lya + l1 + l2
            
        # get gradient w.r.t. parameters
        delta_y = tape.gradient(L_t, self.theta_y)
        # apply optional gradient clipping
        if update_radius > 0.0:
            for p in range(len(delta_y)):
                pdelta = delta_y[p]
                pdelta = tf.clip_by_value(pdelta, -update_radius, update_radius)
                delta_y[p] = pdelta
        # update parameters given derivatives
        self.y_opt.apply_gradients(zip(delta_y, self.theta_y))
        return L_t

    def decode_y_ensemble(self, xi, all_annotator_metadata=None):
        """
            Computes the label distribution given only an item feature vector
            (and model's knowledge of all known annotators with their metadata).
            
            Args:
                xi: Item feature vector [batch_size, xi_dim]
                all_annotator_metadata: All annotator metadata embeddings [num_annotators, meta_dim]
                                       If None, will use dummy metadata
        """
        drop_p = self.drop_p + 0
        self.drop_p = 0.0 # turn off dropout

        z_i = self.encode_i(xi)
        batch_size = xi.shape[0]
        
        # Handle ensemble prediction with metadata
        if all_annotator_metadata is not None:
            # Use provided annotator metadata
            num_annotators = all_annotator_metadata.shape[0]
            
            # Tile the item embedding to match number of annotators
            # [batch_size, lat_i_dim] -> [batch_size * num_annotators, lat_i_dim]
            tiled_z_i = tf.tile(z_i, [num_annotators, 1])
            
            # Tile the metadata to match batch size
            # [num_annotators, meta_dim] -> [batch_size * num_annotators, meta_dim]
            tiled_metadata = tf.tile(all_annotator_metadata, [batch_size, 1])
            
            # Encode annotator metadata
            z_a = self.encode_a(tiled_metadata)
            
            # Combine item and annotator embeddings
            if self.lat_fusion_type == "concat":
                z = self.fx(tf.concat([tiled_z_i, z_a], axis=1))
            else:
                z = self.fx(tiled_z_i + z_a)
            
            z = self.transform(z)
            y_prob, y_logits = self.decode_y(z)
            
            # Reshape to [batch_size, num_annotators, y_dim]
            y_prob = tf.reshape(y_prob, [batch_size, num_annotators, -1])
            
            # Average across annotators to get ensemble prediction
            ensemble_y_prob = tf.reduce_mean(y_prob, axis=1)
            
        else:
            # Fallback: use dummy metadata for ensemble
            # Create dummy metadata with same dimension as expected
            dummy_meta = tf.zeros([batch_size, self.a_dim])
            z_a = self.encode_a(dummy_meta)
            
            if self.lat_fusion_type == "concat":
                z = self.fx(tf.concat([z_i, z_a], axis=1))
            else:
                z = self.fx(z_i + z_a)
            
            z = self.transform(z)
            ensemble_y_prob, y_logits = self.decode_y(z)

        self.drop_p = drop_p # turn dropout back on
        return ensemble_y_prob, y_logits

    def infer_a(self, xi, yi, K, beta, gamma=0.0, is_verbose=False):
        """
            Infer an annotator embedding given only an item feature and label
            distribution vector pair.
        """
        print("WARNING: DO NOT USE THIS! NOT DEBUGGED FOR CONCAT AT THE MOMENT!")
        best_L = None
        batch_size = yi.shape[0]
        z_eps = 0.0 #0.001
        if "elu" in self.act_fx:
            z_eps = 0.001
        # Step 1: encode xi
        z_i = self.encode_i(xi)
        self.z_a = tf.Variable(tf.zeros([batch_size,self.lat_dim]) + z_eps, name="z_a", dtype=tf.float32)
        # Step 2: find za given xi, yi
        for k in range(K):
            with tf.GradientTape(persistent=True) as tape:
                z = self.fx(z_i + self.z_a)
                z = self.transform(z)
                yi_prob, yi_logits = self.decode_yi(z)
                Lyi = D_KL(yi, yi_prob) * self.gamma_i
                Lyi = tf.reduce_sum(Lyi)
                if is_verbose is True:
                    print("k({0}) KL(p(yi)||yi) = {1}".format(k, Lyi))
            # check early halting criterion
            if best_L is not None:
                if Lyi < best_L:
                    best_L = Lyi
                else:
                    break # early stop at this point
            else:
                best_L = Lyi
            d_z_a = tape.gradient(Lyi, self.z_a) # get KL gradient w.r.t. z_a
            self.z_a.assign( self.z_a - d_z_a * beta - self.z_a * gamma) # update latent z_a
        z_a = self.z_a
        return z_a

    def clear(self):
        self.z_i = tf.Variable(tf.zeros([1,self.lat_dim]), name="z_i", dtype=tf.float32)
        self.z_a = tf.Variable(tf.zeros([1,self.lat_dim]), name="z_a", dtype=tf.float32)
