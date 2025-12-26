import { useState } from "react";
import axios from "axios";
import "./App.css";

const BACKEND = "http://127.0.0.1:8003";

export default function App() {
  // ================= IMAGE RESTORATION =================
  const [restoreFile, setRestoreFile] = useState(null);
  const [originalRestoreImg, setOriginalRestoreImg] = useState(null);
  const [restoredImage, setRestoredImage] = useState(null);

  // ================= TAMPERING DETECTION =================
  const [suspectFile, setSuspectFile] = useState(null);
  const [tamperText, setTamperText] = useState("");
  const [suspImg, setSuspImg] = useState(null);
  const [maskImg, setMaskImg] = useState(null);

  // ================= RESTORE IMAGE =================
  const handleRestore = async () => {
    if (!restoreFile) return alert("Please upload an image");

    setOriginalRestoreImg(null);
    setRestoredImage(null);

    const formData = new FormData();
    formData.append("file", restoreFile);

    try {
      const res = await axios.post(`${BACKEND}/restore`, formData);

      if (res.data.error) {
        alert(res.data.error);
        return;
      }

      setOriginalRestoreImg(`data:image/png;base64,${res.data.original_image}`);
      setRestoredImage(`data:image/png;base64,${res.data.restored_image}`);
    } catch (err) {
      alert("Restoration failed. Check backend.");
      console.error(err);
    }
  };

  const downloadImage = () => {
    const link = document.createElement("a");
    link.href = restoredImage;
    link.download = "restored_image.png";
    link.click();
  };

  // ================= TAMPERING DETECTION =================
  const handleDetect = async () => {
    if (!suspectFile) return alert("Please upload a suspected image");

    setTamperText("");
    setSuspImg(null);
    setMaskImg(null);

    const formData = new FormData();
    formData.append("file", suspectFile);

    try {
      const res = await axios.post(`${BACKEND}/detect`, formData);

      if (res.data.error) {
        alert(res.data.error);
        return;
      }

      const { label, fm_confidence, suspect_image, forgery_mask } = res.data;

      // ✅ Final required output text
      setTamperText(`${label} Image (Confidence: ${fm_confidence}%)`);

      setSuspImg(`data:image/png;base64,${suspect_image}`);
      setMaskImg(`data:image/png;base64,${forgery_mask}`);
    } catch (err) {
      alert("Detection failed. Check backend.");
      console.error(err);
    }
  };

  return (
    <div className="container">
      <h1>AI Image Security</h1>

      <div className="grid">
        {/* ================= IMAGE RESTORATION ================= */}
        <div className="card">
          <h2>Image Restoration</h2>
          <p>Restore blurred or degraded images</p>

          <input
            type="file"
            accept="image/*"
            onChange={(e) => setRestoreFile(e.target.files[0])}
          />

          <button onClick={handleRestore}>Restore</button>

          {restoredImage && (
            <>
              <div className="result-grid">
                <div>
                  <h4>Original</h4>
                  <img src={originalRestoreImg} alt="Original" />
                </div>

                <div>
                  <h4>Restored</h4>
                  <img src={restoredImage} alt="Restored" />
                </div>
              </div>

              <button onClick={downloadImage}>Download Restored</button>
            </>
          )}
        </div>

        {/* ================= TAMPERING DETECTION ================= */}
        <div className="card">
          <h2>Tampering Detection</h2>
          <p>Detect image tampering using Hybrid Quantum AI</p>

          <label>Suspected Image</label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setSuspectFile(e.target.files[0])}
          />

          <button onClick={handleDetect}>Detect Tampering</button>

          {tamperText && <h3>{tamperText}</h3>}

          {maskImg && (
            <div className="result-grid">
              <div>
                <h4>Suspected Image</h4>
                <img src={suspImg} alt="Suspected" />
              </div>

              <div>
                <h4>Forgery Mask</h4>
                <img src={maskImg} alt="Forgery Mask" />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
