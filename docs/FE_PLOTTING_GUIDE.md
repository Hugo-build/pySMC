# FE Model Plotting Guide

## Overview
The MCP server now supports plotting FE (Finite Element) models directly in the Streamlit chat interface. The plotting feature generates 3D visualizations of truss structures and displays them inline in the chat.

## Features

### 1. **FE.plot_model** Tool
- **Description**: Generates a 3D plot of an FE model from `nodes.csv` and `elements.csv` files
- **Input**: 
  - `directory` (required): Path to directory containing FE files
  - `show_node_labels` (optional, default=True): Whether to display node labels
- **Output**: Base64-encoded PNG image displayed in chat

### 2. Visualization Details
- **3D Plot**: Interactive-style 3D visualization using matplotlib
- **Elements**: Shown as blue lines with circular markers
- **Nodes**: Labeled with "N{node_id}" in red text (if labels enabled)
- **Axes**: Labeled with X, Y, Z coordinates in millimeters
- **Equal Aspect Ratio**: Ensures proper geometric representation

## Usage in Streamlit App

### Starting the App
```bash
streamlit run app.py
```

Make sure your `.env` file contains:
```
OPENAI_API_KEY=your_api_key_here
MCP_SERVER_SCRIPT=server_fe.py
```

### Example Prompts

1. **Basic Plotting**:
   ```
   Plot the FE model in the 40barTruss directory
   ```

2. **Without Node Labels**:
   ```
   Plot the FE model in examples/40barTruss without node labels
   ```

3. **With Model Information**:
   ```
   Show me information about the FE model in 40barTruss and then plot it
   ```

4. **Exploratory**:
   ```
   Find FE models in the current directory and plot them
   ```

## How It Works

### Backend (server_fe.py)
1. Loads nodes and elements from CSV files
2. Creates a 3D matplotlib figure
3. Plots elements as lines connecting nodes
4. Adds labels and styling
5. Converts plot to base64-encoded PNG
6. Returns encoded image in JSON response

### Frontend (app.py)
1. Receives tool result from MCP server
2. Detects `image_base64` field in response
3. Decodes base64 string to image bytes
4. Displays image using Streamlit's `st.image()`
5. Stores image in message history for persistence

## File Structure

### Required FE Files
Your FE model directory must contain:

**nodes.csv**:
```csv
node_id,x,y,z
1,0.0,0.0,0.0
2,1000.0,0.0,0.0
...
```

**elements.csv**:
```csv
element_id,node_i,node_j,E,A,rho
1,1,2,210000,100,7.85e-6
2,2,3,210000,100,7.85e-6
...
```

## Dependencies

The plotting feature requires:
- `matplotlib>=3.10.7` - For plot generation
- `pillow>=10.0.0` - For image handling in Streamlit
- `numpy>=1.26` - For numerical operations

All dependencies are specified in `pyproject.toml`.

## Integration with Other Tools

The plotting tool works seamlessly with other FE tools:

1. **FE.detect_files**: Find FE models in directories
2. **FE.load_model**: Load and validate model data
3. **FE.get_info**: Get model statistics
4. **FE.plot_model**: Visualize the model (NEW!)
5. **FE.solve_model**: Solve static problems (coming soon)

## Example Workflow

```python
# In chat:
User: "What FE models are available?"
AI: [Calls FE.detect_files]
    "Found 40barTruss directory with FE files"

User: "Tell me about that model"
AI: [Calls FE.get_info]
    "The model has 20 nodes and 40 elements..."

User: "Plot it for me"
AI: [Calls FE.plot_model]
    [Displays 3D visualization of the truss]
```

## Troubleshooting

### Issue: Plot not showing
- **Check**: Ensure `pillow` is installed
- **Solution**: Run `uv pip install pillow` or `pip install pillow`

### Issue: Server connection fails
- **Check**: MCP_SERVER_SCRIPT path in .env
- **Solution**: Set to `server_fe.py` (relative to workspace root)

### Issue: FE files not found
- **Check**: Directory path is correct
- **Solution**: Use FE.detect_files tool first to verify paths

### Issue: Image quality is poor
- **Modify**: DPI setting in server_fe.py line 311
- **Default**: 100 DPI (good balance of quality and size)
- **Higher quality**: Change to 150 or 200 DPI

## Future Enhancements

Planned features:
- [ ] Plot deformed structures with displacement vectors
- [ ] Color elements by stress/strain values
- [ ] Interactive 3D rotation (using plotly)
- [ ] Comparison plots (before/after)
- [ ] Animation of dynamic responses
- [ ] Export plots to file system

## Technical Notes

### Base64 Encoding
Images are base64-encoded to:
- Avoid file system operations
- Simplify client-server communication
- Enable easy embedding in JSON responses
- Support message history persistence

### Performance
- Typical plot generation: 100-500ms
- Image size: ~200-500KB (base64 encoded)
- Suitable for models up to ~1000 elements

For larger models, consider:
- Disabling node labels
- Reducing DPI
- Implementing progressive rendering

## Reference

Based on plotting code from:
- `examples/exp_40barTruss.py` (lines 80-96, 182-258)
- MCP specification: https://modelcontextprotocol.io
- Streamlit docs: https://docs.streamlit.io

## Support

For issues or questions:
1. Check this guide
2. Review `server_fe.py` implementation
3. Test with `examples/40barTruss/` sample data
4. Check Streamlit console for error messages

