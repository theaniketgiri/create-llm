# Bug Fixes Applied to create-llm Scaffolder

## Date: 2025-01-24

## Summary
Fixed critical bugs in the create-llm scaffolder that were causing training failures for users after project generation.

## Bugs Fixed

### 1. **Data Loading Bug in LLMDataset** (CRITICAL)
**Location:** `src/python-dataset-templates.ts`

**Problem:**
- The `_load_data()` method was incorrectly handling 2D tensors
- When `prepare.py` saved data as `[num_examples, seq_length]`, the dataset class tried to re-split it
- This resulted in 0 training examples being loaded, causing training to fail with:
  ```
  ValueError: num_samples should be a positive integer value, but got num_samples=0
  ```

**Fix:**
- Updated `_load_data()` to check tensor dimensions:
  - If 2D: Use directly (already split into examples)
  - If 1D: Split into sequences
- Fixed `_split_into_sequences()` range calculation (added `+ 1` to include final sequence)

**Code Changes:**
```python
def _load_data(self):
    if self.data_path.suffix == '.pt':
        data = torch.load(self.data_path)
        if isinstance(data, torch.Tensor):
            # Check if data is already in correct format (2D tensor)
            if data.dim() == 2:
                # Data is already [num_examples, seq_len]
                return data.tolist()
            elif data.dim() == 1:
                # Data is 1D, need to split into sequences
                return self._split_into_sequences(data)
            else:
                raise ValueError(f"Expected 1D or 2D tensor, got {data.dim()}D")
        return data
```

### 2. **Insufficient Sample Data**
**Location:** `src/scaffolder.ts` - `getSampleData()` method

**Problem:**
- Original sample data was only ~1,700 characters
- Created only 2 training examples
- Too small for meaningful training demonstration

**Fix:**
- Expanded sample data to ~5,000 characters
- Now creates 5 training examples
- Added more comprehensive content including:
  - Detailed explanations of data requirements
  - Multiple example domains (code, literature, technical, etc.)
  - Data preparation steps
  - Tips for better results
  - Advanced techniques section

### 3. **UTF-8 Encoding Issues on Windows**
**Location:** `src/scaffolder.ts` - `getTrainingScriptTemplate()` method

**Problem:**
- Emoji characters (🚀, ✓, etc.) in Python scripts caused encoding errors on Windows
- Error: `UnicodeEncodeError: 'charmap' codec can't encode character`

**Fix:**
- Added UTF-8 encoding setup for Windows console at the start of `train.py`:
```python
# -*- coding: utf-8 -*-
# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
```

## Testing

### Test Results
✅ Project generation successful
✅ Tokenizer training works
✅ Data preparation creates 5 examples
✅ Dataset loads 5 examples correctly (was 0 before)
✅ Training starts successfully
✅ No encoding errors on Windows

### Test Commands
```bash
# Generate project
node dist/index.js test-fixed-llm --template tiny --tokenizer bpe --skip-install

# Train tokenizer
cd test-fixed-llm
python tokenizer/train.py --data data/raw/sample.txt

# Prepare data
python data/prepare.py

# Start training
python training/train.py
```

## Impact

### Before Fix
- Users would encounter immediate failure when trying to train
- Error message: "num_samples should be a positive integer value, but got num_samples=0"
- Windows users would see encoding errors
- Sample data was too small for meaningful demonstration

### After Fix
- Training starts successfully
- Data loads correctly (5 examples)
- No encoding errors on Windows
- Better sample data for demonstration
- Users can immediately see the training pipeline working

## Files Modified

1. `src/python-dataset-templates.ts` - Fixed data loading logic
2. `src/scaffolder.ts` - Enhanced sample data and added UTF-8 encoding

## Deployment

To deploy these fixes:
```bash
npm run build
npm publish  # or your deployment process
```

## User Impact

**Breaking Changes:** None - these are bug fixes only

**User Action Required:** None - fixes are automatically included in newly generated projects

**Existing Projects:** Users with existing projects can manually apply the fixes by:
1. Updating `data/dataset.py` with the new `_load_data()` method
2. Updating `training/train.py` with UTF-8 encoding setup
3. Optionally replacing `data/raw/sample.txt` with more comprehensive sample data

## Notes

- The fixes maintain backward compatibility
- No changes to the public API or CLI interface
- All existing tests should continue to pass
- Consider adding automated tests for these scenarios in the future
