
from base_model import KATSum
from transformers import LongT5ForConditionalGeneration,AutoTokenizer
from kg_embedder import KGEncoder

MODEL_NAME = "google/long-t5-tglobal-base"
NUM_SIDECAR_LAYERS = 4 # [1..12]
DEVICE = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = LongT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

kg_embedder = KGEncoder(
    encoder=base_model.encoder,
    tokenizer=tokenizer,
    hidden_dim=base_model.config.d_model,
    device=DEVICE,
)

custom_model = KATSum(
    base_model=base_model,
    kg_embedder=kg_embedder,
    num_sidecar_layers=NUM_SIDECAR_LAYERS,
    fusion_gate_biases=None, # Can be customized (See KGSidecarLayer)
    device=DEVICE,
)

custom_model.parameter_count()
