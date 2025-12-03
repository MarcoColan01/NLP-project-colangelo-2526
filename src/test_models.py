# test_step1_models.py
import torch
import sys
import os

# Aggiungiamo la cartella corrente al path per importare i moduli
sys.path.append(os.getcwd())
from src.models.teacher import BERTTeacher
from src.models.student import TinyBERTStudent


def test_models():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Hardware check: Using {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM Free: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB")

    # 2. Caricamento Teacher
    try:
        teacher = BERTTeacher().to(device)
        print("✅ Teacher caricato su GPU/CPU")
    except Exception as e:
        print(f"❌ Errore caricamento Teacher: {e}")
        return

    # 3. Caricamento Student
    try:
        student = TinyBERTStudent(teacher.config).to(device)
        print("✅ Student caricato su GPU/CPU")
    except Exception as e:
        print(f"❌ Errore caricamento Student: {e}")
        return

    # 4. Dummy Forward Pass (Simulazione di una frase)
    print("🔄 Test Forward pass (Dummy Input)...")
    batch_size = 2
    seq_len = 32
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    dummy_mask = torch.ones((batch_size, seq_len)).to(device)

    with torch.no_grad():
        t_out = teacher(dummy_input, dummy_mask)
        s_out = student(dummy_input, dummy_mask)

    # 5. Verifica Output
    # Teacher hidden size = 768, Student hidden size = 312
    print(f"   Teacher Output shape: {t_out.logits.shape}") # [2, 32, 30522]
    print(f"   Student Output shape: {s_out.logits.shape}") # [2, 32, 30522]
    
    # Test Proiezione
    hidden_student = s_out.hidden_states[-1] # Ultimo layer
    projected = student.fit_dense(hidden_student)
    print(f"   Projected Student Hidden: {projected.shape}") # Deve essere [2, 32, 768]

    if projected.shape[-1] == 768:
        print("✅ Proiezione dimensionale corretta (312 -> 768)")
    else:
        print("❌ Errore nella proiezione dimensionale")

    if device.type == 'cuda':
        print(f"   VRAM Used Final: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

if __name__ == "__main__":
    test_models()