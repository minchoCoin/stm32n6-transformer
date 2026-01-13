import tensorflow as tf
import numpy as np
# -----------------------------
# Positional Encoding
# -----------------------------
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis].astype(np.float32)
    i = np.arange(d_model)[np.newaxis, :].astype(np.float32)
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return tf.constant(pe, dtype=tf.float32)


class LayerNormSafe(tf.keras.layers.Layer):
    def __init__(self, d_model, epsilon=1e-6):
        super().__init__()
        self.gamma = tf.Variable(tf.ones([d_model]), trainable=True, dtype=tf.float32)
        self.beta = tf.Variable(tf.zeros([d_model]), trainable=True, dtype=tf.float32)
        self.epsilon = tf.constant(epsilon, dtype=tf.float32)

    def call(self, X):
        # X: (batch, seq_len, d_model)
        mean = tf.reduce_mean(X, axis=-1, keepdims=True)              
        diff = X - mean
        var = tf.reduce_mean(diff * diff, axis=-1, keepdims=True)    
       
        inv_std = tf.math.rsqrt(var + self.epsilon)                 
        X_norm = (X - mean) * inv_std
        return X_norm * self.gamma + self.beta


# -----------------------------
# Multi-Head Self-Attention (Dense + Dot)
# -----------------------------
class MultiHeadSelfAttentionDense(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.q_proj = tf.keras.layers.Dense(d_model,use_bias=False)
        self.k_proj = tf.keras.layers.Dense(d_model,use_bias=False)
        self.v_proj = tf.keras.layers.Dense(d_model,use_bias=False)
        self.out_proj = tf.keras.layers.Dense(d_model,use_bias=False)

    def split_heads(self, x):
        # (batch, seq_len, d_model) → (batch, num_heads, seq_len, depth)
        b = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (b, seq_len, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        # (batch, num_heads, seq_len, depth) → (batch, seq_len, d_model)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        b = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        return tf.reshape(x, (b, seq_len, self.d_model))

    def call(self, X):
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        Qh = self.split_heads(Q)
        Kh = self.split_heads(K)
        Vh = self.split_heads(V)


        scale = tf.cast(tf.sqrt(tf.cast(self.depth, tf.float32)), tf.float32)
        Kh = tf.transpose(Kh, [0,1,3,2])
        scores = tf.matmul(Qh, Kh) / scale  # (b, heads, seq, seq)
        attn = tf.nn.softmax(scores, axis=-1)

        context = tf.matmul(attn, Vh)  # (b, heads, seq, depth)
        context_combined = self.combine_heads(context)
        out = self.out_proj(context_combined)
        return out
# -----------------------------
# Feed-Forward Network
# -----------------------------
class FeedForwardSimple(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(d_ff, activation='relu', use_bias=False)
        self.fc2 = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, X):
        return self.fc2(self.fc1(X))

# -----------------------------
# Encoder Block
# -----------------------------
class EncoderBlockSafe(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, use_norm=False):
        super().__init__()
        self.mha = MultiHeadSelfAttentionDense(d_model, num_heads)
        self.ffn = FeedForwardSimple(d_model, d_ff)
        self.use_norm = use_norm
        if use_norm:
            self.ln1 = LayerNormSafe(d_model)
            self.ln2 = LayerNormSafe(d_model)
        else:

            self.ln1 = tf.keras.layers.Lambda(lambda x: x)
            self.ln2 = tf.keras.layers.Lambda(lambda x: x)

    def call(self, X):
        # Attention block
        attn_out = self.mha(X)
        x = X + attn_out  # residual
        if self.use_norm:
            x = self.ln1(x)
        # FFN block
        ffn_out = self.ffn(x)
        x = x + ffn_out  # residual
        if self.use_norm:
            x = self.ln2(x)
        return x

class TransformerEncoderSafe(tf.keras.Model):
    def __init__(self, num_blocks, d_model, num_heads, d_ff, seq_len, num_classes, dropout, use_norm=False):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.pos_enc = positional_encoding(seq_len + 1, d_model)  # CLS 포함
        self.cls_token = self.add_weight(
            shape=(1, 1, d_model),
            initializer="zeros",
            trainable=True,
            name="cls_token"
        )
        self.dropout = dropout
        self.blocks = [EncoderBlockSafe(d_model, num_heads, d_ff, use_norm=use_norm)
                       for _ in range(num_blocks)]
        self.head = tf.keras.layers.Dense(num_classes,activation="softmax",use_bias=False)
        
        self.drop = L.Dropout(self.dropout)
    def call(self, X):


        B = tf.shape(X)[0]

        # 1) CLS token broadcast
        cls = tf.repeat(self.cls_token, repeats=B, axis=0)

        # 2) prepend cls
        X = tf.concat([cls, X], axis=1)      # [B, seq_len+1, d_model]

        # 3) add pos enc
        X = X + self.pos_enc

        # 4) pass through Transformer blocks
        for b in self.blocks:
            X = b(X)
        X=self.drop(X)
        # 5) take CLS token only
        X = X[:, 0, :]   # [B, D]

        logits = self.head(X)
        return logits
    
import kagglehub
from tensorflow.keras import layers as L
# Download latest version
path = kagglehub.dataset_download("imsparsh/flowers-dataset")

print("Path to dataset files:", path)

def build_vit(
    image_size=256,          # 입력 이미지 한 변 크기
    patch_size=16,           # 패치 한 변 크기
    num_classes=10,          # 클래스 수
    dim=768,                 # 토큰 임베딩 차원
    depth=6,                 # Transformer layer 개수
    heads=3,                 # Multi-Head 수
    mlp_dim=384,             # MLP 내부 차원
    dropout=0.1,
    use_norm=False

):
    assert image_size % patch_size == 0, "image_size는 patch_size로 나누어 떨어져야 합니다."
    num_patches = (image_size // patch_size) ** 2


    # WE CHANGE THE INPUT AFTER PATCH EMBEDDING
    inputs = L.Input(shape=(num_patches,patch_size*patch_size*3),batch_size=1)

    x = L.Dense(dim,use_bias=False)(inputs)


    x = TransformerEncoderSafe(d_model=dim, num_heads=heads, d_ff=mlp_dim,seq_len=num_patches,num_blocks=depth, num_classes=num_classes, dropout=dropout, use_norm=use_norm)(x)

   

    return tf.keras.Model(inputs, x, name="TinyViT")


def get_model(img,patch,nclass,d_model,num_blocks,num_heads,d_ff,dropout,use_layernorm):
    

    seq_len = (img//patch)**2
    model = build_vit(img,patch,nclass,d_model,num_blocks,num_heads,d_ff,dropout,use_layernorm)
    
    dummy = tf.random.uniform((1, (img//patch)**2, patch*patch*3))
    _ = model(dummy)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    
    return model,seq_len

def preprocess(x, y):
   
    x = tf.cast(x, tf.float32) / 255.0

    patches = tf.image.extract_patches(
        images=x,
        sizes=[1, GLOBAL_PATCH_SIZE, GLOBAL_PATCH_SIZE, 1],
        strides=[1, GLOBAL_PATCH_SIZE, GLOBAL_PATCH_SIZE, 1],
        rates=[1, 1, 1, 1],
        padding="VALID"
    )

    patches = tf.reshape(patches, (tf.shape(x)[0], -1, patches.shape[-1]))
    
    return patches, y


import glob
import random
import numpy as np
import os
from PIL import Image
import tensorflow as tf

def load_image_as_float(path, img_size):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((img_size, img_size), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr  # (H, W, 3)


def preprocess_for_rep(x):
    # x: numpy array (H,W,3)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = tf.expand_dims(x, 0)                       # (1, IMG, IMG, 3)

    patches = tf.image.extract_patches(
        images=x,
        sizes=[1, GLOBAL_PATCH_SIZE, GLOBAL_PATCH_SIZE, 1],
        strides=[1, GLOBAL_PATCH_SIZE, GLOBAL_PATCH_SIZE, 1],
        rates=[1, 1, 1, 1],
        padding="VALID"
    )

    patches = tf.reshape(patches, (tf.shape(x)[0], -1, patches.shape[-1]))
    

    return patches.numpy()


def representative_data_gen(img_size):
    exts = ("*.jpg","*.jpeg")
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(f"{path}/test", ext)))

    if not img_paths:
        raise FileNotFoundError("No test images found.")

    # representative data 200
    for p in img_paths[:200]:
        raw = load_image_as_float(p, img_size)   # (H,W,3)
        x = preprocess_for_rep(raw)         # (1, num_patches, depth)
        yield [x]

import argparse
GLOBAL_PATCH_SIZE = 16
def main():
    global GLOBAL_PATCH_SIZE
    tf.random.set_seed(42)
    import random
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=224, help="Input image size (default: 224)")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size (default: 16)")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes (default: 5)")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer d_model (default: 128)")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of Transformer blocks (default: 4)")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads (default: 4)")
    parser.add_argument("--d_ff", type=int, default=256, help="Feed-forward network dimension (default: 256)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--use_layernorm", action="store_true", help="Use LayerNorm in Transformer blocks")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--input_type", type=str, default="uint8", help="TFLite input type: float32, int8, uint8")
    parser.add_argument("--output_type", type=str, default="float32", help="TFLite output type: float32, int8, uint8")
    args = parser.parse_args()
    GLOBAL_PATCH_SIZE = args.patch_size
    train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{path}/train",
    image_size=(args.img_size,args.img_size),
    batch_size=1,
    shuffle=True
    )
    model,seqlen = get_model(args.img_size,args.patch_size,args.num_classes,args.d_model,args.num_blocks,args.num_heads,args.d_ff,args.dropout,args.use_layernorm)

    train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    model.fit(train_ds, epochs=args.epochs)

   
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data_gen(args.img_size)

    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    if args.input_type == "int8":
        converter.inference_input_type  = tf.int8
    elif args.input_type == "uint8":
        converter.inference_input_type  = tf.uint8
    else:
        converter.inference_input_type  = tf.float32

    #converter.inference_input_type  = tf.uint8
    if args.output_type == "int8":
        converter.inference_output_type = tf.int8
    elif args.output_type == "uint8":
        converter.inference_output_type = tf.uint8
    else:
        converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    if args.use_layernorm:
        filename=f"custom_vit_ln_im{args.img_size}_attdim{args.d_model}_depth{args.num_blocks}_head{args.num_heads}_ff{args.d_ff}.tflite"
    else:
        filename=f"custom_vit_im{args.img_size}_attdim{args.d_model}_depth{args.num_blocks}_head{args.num_heads}_ff{args.d_ff}.tflite"
    with open(filename, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved: {filename}")

if __name__ == "__main__":
    main()