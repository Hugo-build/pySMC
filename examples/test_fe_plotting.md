# Testing FE Plotting Feature

## Quick Start

1. **Start the Streamlit App**:
```bash
cd /Users/yuma/Library/CloudStorage/OneDrive-UniversitetetiStavanger/UiS_PhD_workingspace/git_devProj/pySMC
streamlit run app.py
```

2. **Try these prompts in the chat**:

### Basic Tests

**Test 1: Simple Plot**
```
Plot the FE model in the 40barTruss directory
```
Expected: Should show a 3D visualization of a 40-bar truss structure with node labels.

**Test 2: Plot from Examples**
```
Plot the FE model in examples/40barTruss
```
Expected: Same result, testing path handling.

**Test 3: Without Labels**
```
Plot the FE model in 40barTruss without showing node labels
```
Expected: Cleaner plot without "N1", "N2", etc. labels.

### Exploratory Tests

**Test 4: Discovery + Plot**
```
Find all FE models in this project and plot the one in 40barTruss
```
Expected: AI should first call FE.detect_files, then FE.plot_model.

**Test 5: Info + Plot**
```
Tell me about the FE model in 40barTruss and show me a plot
```
Expected: AI calls FE.get_info first, then FE.plot_model, explaining the structure.

**Test 6: Compare Directories**
```
Compare the FE models in 40barTruss and examples/40barTruss
```
Expected: AI should load both, compare stats, and potentially plot both.

## What to Look For

### ✅ Success Indicators
- 3D plot appears inline in the chat
- Plot shows blue lines (elements) connecting nodes
- Node labels are visible (if enabled)
- Axes are labeled (X, Y, Z in mm)
- Plot persists when scrolling up in chat history
- No console errors in terminal

### ❌ Failure Modes
- "Image not found" error → Check Pillow installation
- "MCP server failed" → Check server_fe.py is running
- "FE files not found" → Verify path to 40barTruss
- Blank image → Check matplotlib backend setting
- JSON parse error → Check server response format

## Validation Checklist

- [ ] Plot displays correctly in chat
- [ ] Node labels appear when requested
- [ ] Node labels hidden when disabled
- [ ] Plot persists in message history
- [ ] Multiple plots can be shown in same chat
- [ ] Equal aspect ratio maintained
- [ ] All nodes and elements visible
- [ ] Title shows correct node/element count

## Advanced Tests

**Test 7: Multiple Plots**
```
Plot the 40barTruss model twice - once with labels and once without
```
Expected: Two plots appear, demonstrating history persistence.

**Test 8: Error Handling**
```
Plot the FE model in nonexistent_directory
```
Expected: Graceful error message, no crash.

## Behind the Scenes

When you send "Plot the FE model in 40barTruss":

1. **Streamlit App** (`app.py`):
   - Sends message to OpenAI API
   - OpenAI decides to call `FE.plot_model` tool
   - App receives tool call instruction

2. **MCP Server** (`server_fe.py`):
   - Receives `FE.plot_model` command
   - Loads nodes.csv and elements.csv
   - Creates matplotlib 3D figure
   - Converts figure to PNG bytes
   - Encodes as base64 string
   - Returns JSON with image_base64 field

3. **Streamlit App** (continued):
   - Receives base64 response
   - Decodes to image bytes
   - Displays using st.image()
   - Saves to message history

## Troubleshooting

### Issue: "No module named 'PIL'"
**Solution**:
```bash
uv pip install pillow
# or
pip install pillow
```

### Issue: "MCP server not responding"
**Check**:
```bash
# Test server directly
python server_fe.py
```
Should start without errors (will wait for JSON-RPC input).

### Issue: "Directory not found"
**Debug**:
```
What's the current working directory?
```
Then use absolute paths or correct relative paths.

### Issue: Plot is too small/large
**Adjust** in `server_fe.py` line 249:
```python
fig = plt.figure(figsize=(12, 9))  # Increase or decrease these numbers
```

### Issue: Poor image quality
**Adjust** in `server_fe.py` line 311:
```python
plt.savefig(buffer, format='png', dpi=100, ...)  # Increase DPI (e.g., 150, 200)
```

## Performance Metrics

Typical timings on standard hardware:
- Tool call initiation: <50ms
- FE file loading: 10-50ms
- Plot generation: 100-300ms
- Base64 encoding: 50-100ms
- Image display: <50ms
- **Total**: ~300-500ms

## Sample Output

When successful, you should see output like:

```
User: Plot the FE model in 40barTruss
