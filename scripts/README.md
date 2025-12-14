# Scripts / Utility Tools

Herramientas auxiliares para trabajar con modelos entrenados.

## 📁 Contenido

```
scripts/
├── sample.py          # Generar texto desde un modelo entrenado
├── merge_lora.py      # Mergear adaptadores LoRA en modelo base
└── test_lora.py       # Comparar modelo base vs LoRA fine-tuned
```

---

## 🎯 sample.py - Generación de Texto

Genera texto desde un modelo entrenado (GPT estándar o con LoRA).

### Uso Básico

```bash
# Generar desde modelo en models/
python scripts/sample.py --checkpoint models/gpt-38M-scientific-pretrain/ckpt.pt

# Con prompt personalizado
python scripts/sample.py \
    --checkpoint models/gpt-38M-scientific-pretrain/ckpt.pt \
    --prompt "In this paper we present" \
    --num_samples 3 \
    --max_tokens 200
```

### Parámetros

```
--checkpoint     Path al checkpoint (.pt) (required)
--prompt         Texto inicial (default: "\n")
--num_samples    Número de muestras a generar (default: 1)
--max_tokens     Tokens a generar por muestra (default: 100)
--temperature    Temperatura de sampling (default: 0.8)
                 Mayor = más aleatorio, menor = más determinista
--top_k          Top-k sampling (default: 200)
--device         Device: cpu o cuda (default: cpu)
--seed           Semilla random (default: 1337)
```

### Ejemplos

**Modelo científico:**
```bash
python scripts/sample.py \
    --checkpoint models/gpt-38M-scientific-pretrain/ckpt.pt \
    --prompt "Abstract: " \
    --max_tokens 150 \
    --temperature 0.7
```

**Modelo LoRA fine-tuned:**
```bash
python scripts/sample.py \
    --checkpoint models/gpt-38M-tinystories-to-scientific-lora/ckpt.pt \
    --prompt "In this study, we investigate" \
    --num_samples 5
```

**Sampling creativo (alta temperatura):**
```bash
python scripts/sample.py \
    --checkpoint models/gpt-38M-scientific-pretrain/ckpt.pt \
    --temperature 1.2 \
    --top_k 50
```

---

## 🔀 merge_lora.py - Mergear LoRA

Combina adaptadores LoRA con el modelo base para crear un modelo estándar (sin overhead de LoRA).

### ¿Por qué mergear?

**Modelo con LoRA:**
- Checkpoint pequeño (~5-10 MB de adaptadores)
- Requiere cargar base + adaptadores
- Ligero overhead en inferencia

**Modelo mergeado:**
- Checkpoint completo (~150 MB para 38M params)
- Carga directa, sin dependencias
- Inferencia más rápida (sin capa extra de LoRA)

### Uso

```bash
# Mergear modelo LoRA
python scripts/merge_lora.py \
    --lora_checkpoint models/gpt-38M-tinystories-to-scientific-lora/ckpt.pt \
    --output_dir models/gpt-38M-tinystories-to-scientific-merged \
    --device cpu
```

### Parámetros

```
--lora_checkpoint    Path al checkpoint LoRA (required)
--output_dir         Directorio de salida para modelo mergeado (required)
--device             Device: cpu o cuda (default: cpu)
```

### Flujo Típico

```bash
# 1. Fine-tune con LoRA
python train.py config/finetune_lora.py

# 2. Mergear para deployment
python scripts/merge_lora.py \
    --lora_checkpoint models/gpt-38M-tinystories-to-scientific-lora/ckpt.pt \
    --output_dir models/gpt-38M-tinystories-to-scientific-merged

# 3. Usar modelo mergeado
python scripts/sample.py \
    --checkpoint models/gpt-38M-tinystories-to-scientific-merged/ckpt.pt
```

---

## 🔬 test_lora.py - Comparación de Modelos

Compara generación de texto entre modelo base y modelo LoRA fine-tuned.

### Uso

```bash
python scripts/test_lora.py \
    --base_checkpoint models/gpt-38M-tinystories-pretrain/ckpt.pt \
    --lora_checkpoint models/gpt-38M-tinystories-to-scientific-lora/ckpt.pt \
    --prompt "Once upon a time" \
    --max_tokens 100
```

### Parámetros

```
--base_checkpoint    Path al modelo base (required)
--lora_checkpoint    Path al modelo LoRA (required)
--prompt             Texto inicial (default: "Once upon a time")
--max_tokens         Tokens a generar (default: 100)
--temperature        Temperatura de sampling (default: 0.8)
--device             Device: cpu o cuda (default: cpu)
```

### Ejemplo de Salida

```
======================================================================
BASE MODEL (TinyStories):
======================================================================
Once upon a time there was a little girl named Lily. She loved to play
outside in the sun. One day she saw a big red ball...

======================================================================
LORA FINE-TUNED MODEL (Scientific):
======================================================================
Once upon a time, the field of machine learning was primarily focused
on supervised learning approaches. Recent advances in self-supervised...

======================================================================
COMPARISON COMPLETE
======================================================================
```

---

## 🛠️ Notas Técnicas

### Path Handling

Todos los scripts en `scripts/` agregan automáticamente el directorio raíz al `sys.path`:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

Esto permite importar módulos del root (`model.py`, `checkpoint.py`, etc.) sin problemas.

### Checkpoints Compatibles

Los scripts funcionan con:
- ✅ Checkpoints estándar (pretraining desde scratch)
- ✅ Checkpoints LoRA (PEFT)
- ✅ Checkpoints mergeados
- ✅ Formatos: `.pt` (PyTorch) y `.safetensors`

### Performance Tips

**Para generación rápida:**
- Usa `--temperature 0.5` (más determinista, menos cómputo)
- Limita `--max_tokens` a lo necesario
- Usa `--device cuda` si tienes GPU

**Para generación creativa:**
- Usa `--temperature 1.0-1.2` (más variedad)
- Ajusta `--top_k` (50-200, menor = más conservador)
- Genera múltiples muestras con `--num_samples`

---

## 📚 Ejemplos Completos

### Pipeline Completo: Pretrain → LoRA → Merge → Sample

```bash
# 1. Pretrain base model
python train.py config/train_tinystories.py

# 2. LoRA fine-tune to scientific domain
python train.py config/finetune_lora.py

# 3. Test comparison
python scripts/test_lora.py \
    --base_checkpoint models/gpt-38M-tinystories-pretrain/ckpt.pt \
    --lora_checkpoint models/gpt-38M-tinystories-to-scientific-lora/ckpt.pt

# 4. Merge for deployment
python scripts/merge_lora.py \
    --lora_checkpoint models/gpt-38M-tinystories-to-scientific-lora/ckpt.pt \
    --output_dir models/gpt-38M-tinystories-to-scientific-merged

# 5. Generate scientific text
python scripts/sample.py \
    --checkpoint models/gpt-38M-tinystories-to-scientific-merged/ckpt.pt \
    --prompt "Abstract: In this paper we" \
    --num_samples 3 \
    --max_tokens 200 \
    --temperature 0.7
```

---

## ❓ FAQ

**Q: ¿Por qué están en scripts/ y no en root?**
A: Para mantener el root limpio. Solo `train.py` y módulos core (`model.py`, `checkpoint.py`) deben estar en root.

**Q: ¿Puedo usar estos scripts con modelos de otros proyectos?**
A: Solo si usan la misma arquitectura GPT y formato de checkpoint. Para otros modelos, necesitarían adaptación.

**Q: ¿Cómo elijo la temperatura?**
A:
- 0.1-0.5: Muy determinista (buen para QA, código)
- 0.6-0.9: Balanceado (buen default)
- 1.0-1.5: Creativo (buen para historias, ideas)

**Q: ¿Qué hace top_k?**
A: Limita el sampling a los K tokens más probables. Menor = más conservador.

---

## 🔗 Ver También

- `../TRAINING_GUIDE.md` - Guía completa de entrenamiento
- `../models/README.md` - Convenciones de naming de modelos
- `../config/` - Configuraciones de entrenamiento
