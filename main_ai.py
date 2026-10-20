from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from gradio_client import Client, handle_file
import os
import tempfile

app = FastAPI(title="Image to 3D Studio")

PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image to 3D Studio</title>
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"></script>
<style>
body { font-family: Arial; background:#111827; color:white;
text-align:center; padding:20px; }
button,input { padding:12px; margin:12px; max-width:95%; }
model-viewer { width:100%; height:350px; background:#1f2937; }
a { color:#93c5fd; }
</style>
</head>
<body>
<h1>My 3D Studio</h1>
<p>Select a picture and generate your 3D model.</p>
<input type="file" id="photo" accept="image/*">
<br>
<button id="go" onclick="generate3D()">Generate 3D Model</button>
<p id="status">Ready</p>
<model-viewer id="viewer" camera-controls auto-rotate
  shadow-intensity="1" style="display:none"></model-viewer>
<p><a id="download" style="display:none">Download 3D model</a></p>
<script>
async function generate3D() {
  const photo = document.getElementById('photo');
  const status = document.getElementById('status');
  const button = document.getElementById('go');
  if (!photo.files.length) {
    status.textContent = 'Please select an image first.';
    return;
  }
  button.disabled = true;
  status.textContent = 'Generating model. This may take several minutes...';
  try {
    const data = new FormData();
    data.append('file', photo.files[0]);
    const response = await fetch('/generate', {
      method: 'POST', body: data
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const viewer = document.getElementById('viewer');
    viewer.src = url;
    viewer.style.display = 'block';
    const link = document.getElementById('download');
    link.href = url;
    link.download = 'my_model.glb';
    link.style.display = 'inline';
    status.textContent = 'Your 3D model is ready!';
  } catch (error) {
    status.textContent = 'Generation failed: ' + error.message;
  } finally {
    button.disabled = false;
  }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    with open(os.path.join(os.path.dirname(__file__), "studio.html"), encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")

    suffix = os.path.splitext(file.filename or "photo.png")[1] or ".png"
    input_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            input_path = temp.name
            temp.write(await file.read())

        client = Client("https://microsoft-trellis-2.hf.space")
        client.predict(api_name="/start_session")

        client.predict(
            handle_file(input_path),
            0, "1024", 7.5, 0.7, 12, 5.0,
            7.5, 0.5, 12, 3.0,
            1.0, 0.0, 12, 3.0,
            api_name="/image_to_3d"
        )

        result = client.predict(
            300000, 2048, api_name="/extract_glb"
        )

        candidates = result if isinstance(result, (tuple, list)) else [result]
        model_path = None

        for item in candidates:
            if isinstance(item, dict):
                path = item.get("path") or item.get("name")
            elif isinstance(item, str):
                path = item
            else:
                path = None

            if path and os.path.isfile(path) and path.lower().endswith(".glb"):
                model_path = path
                break

        if not model_path:
            raise RuntimeError("The AI service did not return a GLB file.")

        return FileResponse(
            model_path,
            media_type="model/gltf-binary",
            filename="my_model.glb"
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="AI generation failed: " + str(exc)
        )
    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)
