import sounddevice as sd
import numpy as np
import whisper
import torch
import queue
import sys
import re
from word2number import w2n
from inference import predict_transaction

# --- Configuration ---
SAMPLE_RATE = 16000 # Whisper expects 16kHz
CHANNELS = 1
# small model improves transcription accuracy significantly over base with acceptable CPU cost.
WHISPER_MODEL_SIZE = "small"

# Global Correction Maps
MERCHANT_CORRECTIONS = {
    "swingy": "swiggy",
    "sugui": "swiggy",
    "swigy": "swiggy",
    "amazone": "amazon",
    "uber ridee": "uber ride"
}

def apply_merchant_corrections(text):
    """
    STT mishears common merchant names.
    Correcting before ML inference improves classification accuracy.
    """
    # Token-based correction prevents accidental substring replacement.
    words = text.split()
    corrected_words = []
    
    for word in words:
        clean_word = word.lower().strip(".,!?")
        if clean_word in MERCHANT_CORRECTIONS:
            corrected_words.append(MERCHANT_CORRECTIONS[clean_word])
        else:
            corrected_words.append(word)
            
    return " ".join(corrected_words)

def record_until_enter(fs=SAMPLE_RATE):
    print("🎙 Recording... Press Enter to stop.")
    
    q = queue.Queue()
    
    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    # Start the stream
    with sd.InputStream(samplerate=fs, channels=CHANNELS, dtype='int16', callback=callback):
        input() # Wait for Enter
        
    print("⏳ Processing...")
    
    # Collect all chunks
    data = []
    while not q.empty():
        data.append(q.get())
        
    if not data:
        return np.array([], dtype='int16')
        
    # Concatenate into single array
    recording = np.concatenate(data, axis=0)
    return recording

def normalize_number_words(text):
    """
    Converts number words to digits (e.g., 'three hundred' -> '300').
    Uses safer chunk-based parsing to avoid corrupting normal words.
    """
    try:
        # Optimize: Only attempt conversion if number words are present
        number_words = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 
                        'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 
                        'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 
                        'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion'}
        
        lower_text = text.lower()
        
        # Use word-boundary matching to avoid false positives like "someone".
        number_pattern = r'\b(' + '|'.join(number_words) + r')\b'
        if not re.search(number_pattern, lower_text):
            return text

        # Attempt safe logical replacement
        # Phrase-level conversion prevents incorrect token replacement.
        # w2n is greedy, so we only target phrases that look like numbers.
        words = text.split()
        new_words = []
        i = 0
        while i < len(words):
            word = words[i]
            # Look ahead for multi-word numbers "three hundred fifty"
            chunk = [word]
            j = i + 1
            while j < len(words) and words[j].lower() in number_words:
                chunk.append(words[j])
                j += 1
            
            chunk_str = " ".join(chunk)
            
            # Safe conversion prevents partial token corruption
            try:
                 # Check if the chunk is convertible
                 # Only convert if it's in our number word set or w2n accepts it
                 if any(w.lower() in number_words for w in chunk):
                     val = w2n.word_to_num(chunk_str)
                     new_words.append(str(val))
                     i = j # Skip processed words
                     continue
            except ValueError:
                pass
            
            # If no conversion, append original word
            new_words.append(word)
            i += 1
            
        return " ".join(new_words)
    except Exception as e:
        print(f"Warning: Number normalization failed: {e}")
        return text

def main():
    print("--- 🗣️  Voice Transaction Entry System (Optimized) ---")
    
    # 2️⃣ Load Model Once Strategy
    print(f"⏳ Loading Whisper Model ({WHISPER_MODEL_SIZE})...")
    
    # Optional GPU Support
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device.upper()}")
    
    try:
        model = whisper.load_model(WHISPER_MODEL_SIZE, device=device) 
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Error loading Whisper model: {e}")
        return

    while True:
        try:
            input("\nPress Enter to start recording (or Ctrl+C to exit)...")
            
            # 1. Record
            audio_int16 = record_until_enter()
            
            # Silence detection before normalization prevents amplifying noise.
            # Dual-threshold check (max & mean) on raw int16 data.
            # int16 max is 32768, so threshold needs scaling or check raw values.
            # Let's convert to raw float for check but NOT normalize amplitude yet.
            audio_raw = audio_int16.flatten().astype(np.float32) / 32768.0
            
            if np.max(np.abs(audio_raw)) < 0.005 or np.mean(np.abs(audio_raw)) < 0.001:
                print("⚠️  No speech detected. Please try again.")
                continue

            # Normalize amplitude (only if speech detected)
            # Prevents clipping and volume variation issues.
            max_val = np.max(np.abs(audio_raw))
            if max_val > 0:
                audio_data = audio_raw / max_val
            else:
                audio_data = audio_raw
            
            # 2. Transcribe (Direct from Memory)
            print("⏳ Transcribing...")
            # fp16=False for CPU compatibility. 
            # temperature=0.0 ensures deterministic output. beam_size=5 improves stability and reduces hallucinations.
            result = model.transcribe(
                audio_data, 
                fp16=False,
                temperature=0.0,
                beam_size=5,
                best_of=5,
                language='en'
            )
            text = result["text"].strip()
            
            if not text:
                print("⚠️  No speech detected.")
                continue
            
            # 3. Post-Process
            text = normalize_number_words(text)
            text = apply_merchant_corrections(text)

            # Whisper may output formatted numbers ("3,000") or split digits ("3 000")
            # Merging prevents incorrect multi-candidate detection
            text = re.sub(r'(?<=\d)\s+(?=\d)', '', text) # Merge split digits
            text = re.sub(r'(\d),(?=\d{3})', r'\1', text) # Remove comma formatting
            
            print(f"📝 You said: \"{text}\"")
            
            # 4. Predict
            cat, conf, amount = predict_transaction(text)

            # Low-confidence predictions usually result from unclear audio or STT errors.
            if conf < 0.30:
                print(f"⚠️  Low confidence ({conf:.2f}). Please repeat clearly.")
                continue

            # Prevent accidental zero extraction from malformed audio
            if amount == 0:
                # Discard zero only if likely caused by malformed split digits (e.g. "3 000")
                # and not an explicit "0" in the text.
                numbers_in_text = re.findall(r'\d+', text)
                if len(numbers_in_text) > 1 and '0' not in numbers_in_text:
                    amount = None

            amount_str = f"₹{amount}" if amount is not None else "Not found"
            
            # 5. Output
            print("-" * 40)
            print(f"📂 Category: {cat}")
            print(f"📊 Confidence: {conf:.2f}")
            print(f"💰 Amount: {amount_str}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
