import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import {
  Upload,
  FileText,
  X,
  CheckCircle,
  Loader,
  AlertCircle,
} from "lucide-react";

export default function PdfUpload() {
  const [files, setFiles] = useState([]);
  const [processing, setProcessing] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    const newFiles = acceptedFiles.map((file) => ({
      file,
      id: crypto.randomUUID(),
      name: file.name,
      size: (file.size / 1024 / 1024).toFixed(2),
      status: "uploaded", // uploaded | processing | completed | error
      progress: 0,
    }));
    setFiles((prev) => [...prev, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    multiple: true,
  });

  const removeFile = (id) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const analyzeFiles = () => {
    setProcessing(true);
    // Simulate AI analysis with progress
    const uploadedFiles = files.filter((f) => f.status === "uploaded");
    uploadedFiles.forEach((f, idx) => {
      setTimeout(() => {
        setFiles((prev) =>
          prev.map((file) =>
            file.id === f.id ? { ...file, status: "processing", progress: 30 } : file
          )
        );
      }, idx * 500);

      setTimeout(() => {
        setFiles((prev) =>
          prev.map((file) =>
            file.id === f.id ? { ...file, progress: 70 } : file
          )
        );
      }, idx * 500 + 1000);

      setTimeout(() => {
        setFiles((prev) =>
          prev.map((file) =>
            file.id === f.id
              ? { ...file, status: "completed", progress: 100 }
              : file
          )
        );
        if (idx === uploadedFiles.length - 1) {
          setProcessing(false);
        }
      }, idx * 500 + 2000);
    });
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "uploaded":
        return <FileText size={18} className="text-blue-400" />;
      case "processing":
        return <Loader size={18} className="text-yellow-400 spin" />;
      case "completed":
        return <CheckCircle size={18} className="text-green-400" />;
      case "error":
        return <AlertCircle size={18} className="text-red-400" />;
      default:
        return null;
    }
  };

  const getStatusLabel = (status) => {
    switch (status) {
      case "uploaded":
        return "Ready";
      case "processing":
        return "Analyzing...";
      case "completed":
        return "Analyzed";
      case "error":
        return "Error";
      default:
        return "";
    }
  };

  const uploadedCount = files.filter((f) => f.status === "uploaded").length;
  const completedCount = files.filter((f) => f.status === "completed").length;

  return (
    <div className="pdf-upload-panel">
      <div className="panel-header">
        <div className="panel-header-icon">
          <Upload size={22} />
        </div>
        <div>
          <h2>Upload Project Documents</h2>
          <p className="panel-subtitle">
            Upload PDFs for AI-powered risk analysis
          </p>
        </div>
      </div>

      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? "dropzone-active" : ""}`}
      >
        <input {...getInputProps()} />
        <div className="dropzone-content">
          <div className="dropzone-icon">
            <Upload size={40} />
          </div>
          <p className="dropzone-title">
            {isDragActive ? "Drop PDFs here..." : "Drag & drop PDFs here"}
          </p>
          <p className="dropzone-subtitle">
            or click to browse your files
          </p>
          <span className="dropzone-badge">PDF only • Max 50MB per file</span>
        </div>
      </div>

      {files.length > 0 && (
        <>
          <div className="file-list-header">
            <h3>
              Uploaded Files{" "}
              <span className="file-count">{files.length}</span>
            </h3>
            {completedCount > 0 && (
              <span className="completed-badge">
                <CheckCircle size={14} /> {completedCount} analyzed
              </span>
            )}
          </div>
          <div className="file-list">
            {files.map((f) => (
              <div key={f.id} className={`file-item file-${f.status}`}>
                <div className="file-item-left">
                  {getStatusIcon(f.status)}
                  <div className="file-info">
                    <span className="file-name">{f.name}</span>
                    <span className="file-meta">
                      {f.size} MB • {getStatusLabel(f.status)}
                    </span>
                  </div>
                </div>
                <button
                  className="file-remove"
                  onClick={() => removeFile(f.id)}
                  title="Remove file"
                >
                  <X size={16} />
                </button>
                {f.status === "processing" && (
                  <div className="file-progress">
                    <div
                      className="file-progress-bar"
                      style={{ width: `${f.progress}%` }}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>

          <button
            className="analyze-btn"
            onClick={analyzeFiles}
            disabled={processing || uploadedCount === 0}
          >
            {processing ? (
              <>
                <Loader size={18} className="spin" /> Analyzing...
              </>
            ) : (
              <>
                <FileText size={18} /> Analyze {uploadedCount > 0 ? `${uploadedCount} File${uploadedCount > 1 ? "s" : ""}` : "Files"}
              </>
            )}
          </button>
        </>
      )}

      <div className="upload-tips">
        <h4>Supported Documents</h4>
        <ul>
          <li>Project plans & schedules</li>
          <li>Sprint reports & retrospectives</li>
          <li>Resource allocation sheets</li>
          <li>Risk assessment documents</li>
          <li>Meeting notes & status updates</li>
        </ul>
      </div>
    </div>
  );
}
