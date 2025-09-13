# CQG-MBQA Framework Modification Guidelines

## 🎯 Core Principle
**Minimal Impact, Maximum Stability**: Always prioritize framework integrity over extensive modifications.

## 📋 Modification Standards

### ✅ **DO: Defensive Programming**
- Add **minimal safety checks** only where critical errors occur
- Use **early returns/continues** to skip problematic data
- **Log warnings** for debugging without changing core logic
- **Preserve original behavior** for normal cases

### ❌ **DON'T: Extensive Rewrites** 
- Avoid changing core algorithms or data structures
- Don't modify working code "for improvement"
- Don't add complex error handling where simple checks suffice
- Don't change APIs or interfaces unnecessarily

## 🛠️ **Current Modifications Log**

### 1. **OpenAI Response Length Fix** ✅ **GOOD**
```python
# Location: cqg.py:488-492
# Issue: GPT-5-mini occasionally returns 20 responses instead of 10
# Fix: Minimal truncation at data ingestion point
parsed = parse_response(response["response"]["body"]["choices"][0]["message"]["content"])
initial_filtering_responses[i]["positives"][j] = parsed[:10] if len(parsed) > 10 else parsed
```
**Impact**: ✅ Minimal, surgical fix at the source

### 2. **Empty Cluster Handling** ✅ **ACCEPTABLE**
```python
# Location: cqg.py:512-514
# Issue: GPT-5-mini refuses to generate questions for some clusters
# Fix: Skip empty clusters with warning
if i not in questions or not questions[i]:
    logger.warning(f"Cluster {i} has no questions, skipping...")
    continue
```
**Impact**: ⚠️ Slightly intrusive but necessary for GPT-5-mini compatibility

### 3. **MBQA Cluster Safety Check** ✅ **GOOD**
```python
# Location: mbqa.py:147-148
# Issue: MBQA expects all clusters to exist from CQG
# Fix: Skip missing clusters in hard negative sampling
if closest_cluster_id not in selected_docs_in_clusters:
    continue
```
**Impact**: ✅ Minimal, preserves MBQA functionality

### 4. **Final Questions Generation Safety** ⚠️ **TOO COMPLEX**
```python
# Location: cqg.py:609-615
# Issue: Final generation iterates over non-existent clusters
# Fix: Multiple safety checks and type corrections
```
**Impact**: ❌ **Over-engineered** - could be simplified

## 🎯 **Future Modification Protocol**

### **Step 1: Analyze**
- Is this a **data issue** or **code issue**?
- Can we fix it at the **source** (API response) vs downstream?
- What's the **minimum change** needed?

### **Step 2: Implement**
- **One-line fixes** preferred over multi-line solutions
- **Early exit patterns**: `if problem: continue/return`
- **Preserve original logic** for 99% of cases
- **Log don't crash**: warn about anomalies but keep running

### **Step 3: Test**
- Ensure **zero impact** on normal operation
- Test with **both clean and problematic data**
- Verify **backward compatibility**

## 📝 **Code Examples**

### ✅ **Good: Minimal Safety Check**
```python
# Before: crashes on edge case
result = data[key]

# After: handles edge case minimally  
if key not in data:
    logger.warning(f"Missing key {key}, skipping...")
    continue
result = data[key]
```

### ❌ **Bad: Over-Engineering**
```python
# Before: works for normal cases
result = process_data(data)

# After: unnecessary complexity
try:
    if data and isinstance(data, dict) and validate_structure(data):
        result = process_data_with_fallback(data)
    else:
        result = handle_edge_cases(data)
except Exception as e:
    result = default_fallback(e)
```

## 🚨 **Emergency Exception**
Only break these rules when:
- **Framework completely broken** for a significant use case
- **No minimal fix possible**
- **User explicitly requests major changes**

## 📊 **Modification Review Checklist**
- [ ] Fixes a **real crash/error** (not just "improvement")
- [ ] Uses **≤3 lines** of added code per fix location  
- [ ] **Zero impact** on normal operation
- [ ] **Logs warnings** for debugging
- [ ] **Preserves all original functionality**
- [ ] **No API/interface changes**

---

**Remember**: The framework worked before - we're just making it more robust against edge cases, not "improving" it.