import axios from "axios";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
});

export const setAuthToken = (token) => {
  if (token) {
    api.defaults.headers.common.Authorization = `Bearer ${token}`;
  } else {
    delete api.defaults.headers.common.Authorization;
  }
};

let unauthorizedHandler = null;
export const setUnauthorizedHandler = (handler) => {
  unauthorizedHandler = handler;
};

const notifyUnauthorized = () => {
  if (api.defaults.headers.common.Authorization && unauthorizedHandler) unauthorizedHandler();
};

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const url = error?.config?.url || "";
    if (error?.response?.status === 401 && !url.startsWith("/auth/")) notifyUnauthorized();
    return Promise.reject(error);
  }
);

// FastAPI trả `detail` là chuỗi (HTTPException) hoặc MẢNG (lỗi validation 422). Render thẳng mảng
// object trong JSX làm React crash -> trắng trang, nên luôn đổi về chuỗi.
const formatDetail = (detail) => {
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => d?.msg || JSON.stringify(d)).join("; ");
  }
  return null;
};

export const getErrorMessage = (err, fallback) => {
  if (err && !err.response && err.request) {
    return (
      `Cannot reach the server at ${api.defaults.baseURL || "(VITE_API_BASE_URL is not set)"}. ` +
      "Please check that the backend is running."
    );
  }
  return formatDetail(err?.response?.data?.detail) || fallback;
};

export const registerUser = (data) =>
  api.post("/auth/register", data);

export const loginUser = (identifier, password) =>
  api.post("/auth/login", { identifier, password });

export const forgotPassword = (email) =>
  api.post("/auth/forgot-password", { email });

export const verifyResetCode = (email, code) =>
  api.post("/auth/verify-reset-code", { email, code });

export const resetPassword = (resetToken, newPassword, confirmPassword) =>
  api.post("/auth/reset-password", {
    reset_token: resetToken,
    new_password: newPassword,
    confirm_password: confirmPassword,
  });

export const getMe = () => api.get("/auth/me");

export const updateProfile = (data) => api.patch("/auth/me", data);

export const changePassword = (currentPassword, newPassword, confirmPassword) =>
  api.post("/auth/change-password", {
    current_password: currentPassword,
    new_password: newPassword,
    confirm_password: confirmPassword,
  });

export const uploadDocument = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/documents/upload", form);
};

export const getDocumentStatus = (documentId) =>
  api.get(`/documents/${documentId}/status`);

export const askQuestion = (question, documentId = null, topK = 4) =>
  api.post("/chat/query", {
    question,
    document_id: documentId,
    top_k: topK,
  });

export const generateQuiz = (topic, numQuestions = 10, documentId = null) =>
  api.post("/quiz/generate", {
    topic,
    num_questions: numQuestions,
    document_id: documentId,
  });

export const submitQuizResult = (quizId, result) =>
  api.post(`/quiz/${quizId}/submit`, result);

// Streams the RAG pipeline as Server-Sent Events so the UI can show each real
// step ("Searching your document", ...). axios cannot stream in the browser,
// so this uses fetch. Calls onEvent({type: "step"|"result"|"error", ...}).
const parseSseFrame = (frame) => {
  const data = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).replace(/^ /, ""))
    .join("\n");
  if (!data) return null;
  try {
    return JSON.parse(data);
  } catch {
    console.warn("Ignoring malformed SSE frame:", data.slice(0, 120));
    return null;
  }
};

// options: { topK, history: [{role, content}], signal: AbortSignal }
// Truyền `signal` để huỷ giữa chừng (nút "Dừng"): server thấy kết nối đóng và dừng sinh câu trả lời.
export const askQuestionStream = async (
  question,
  documentId,
  onEvent,
  { topK = 4, history = [], signal } = {}
) => {
  const response = await fetch(`${api.defaults.baseURL || ""}/chat/query/stream`, {
    method: "POST",
    signal,
    headers: {
      "Content-Type": "application/json",
      ...(api.defaults.headers.common.Authorization
        ? { Authorization: api.defaults.headers.common.Authorization }
        : {}),
    },
    body: JSON.stringify({ question, document_id: documentId, top_k: topK, history }),
  });

  if (response.status === 401) notifyUnauthorized();
  if (!response.ok || !response.body) {
    let detail = "An error occurred while connecting to the AI assistant.";
    try {
      detail = formatDetail((await response.json()).detail) || detail;
    } catch {

    }
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const flush = (frames) => {
    for (const frame of frames) {
      const event = parseSseFrame(frame);
      if (event) onEvent(event);
    }
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, "\n");
    const frames = buffer.split("\n\n");
    buffer = frames.pop();
    flush(frames);
  }
  buffer += decoder.decode();
  if (buffer.trim()) flush([buffer]); // frame cuối không có dòng trống kết thúc
};
