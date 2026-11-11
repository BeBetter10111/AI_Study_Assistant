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

// Gọi khi một request ĐÃ ĐĂNG NHẬP nhận 401: token hết hạn, hoặc bị vô hiệu hoá sau khi đổi
// mật khẩu. App đăng ký handler để đưa người dùng về màn hình đăng nhập thay vì để họ kẹt
// với những request lỗi liên tục.
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
    // 401 của chính các endpoint /auth/* (sai mật khẩu...) là lỗi nhập liệu bình thường.
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

// Quên mật khẩu: email -> mã UUID trong email -> mật khẩu mới
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
export const askQuestionStream = async (question, documentId, onEvent, topK = 4) => {
  const response = await fetch(`${api.defaults.baseURL || ""}/chat/query/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(api.defaults.headers.common.Authorization
        ? { Authorization: api.defaults.headers.common.Authorization }
        : {}),
    },
    body: JSON.stringify({ question, document_id: documentId, top_k: topK }),
  });

  if (response.status === 401) notifyUnauthorized();
  if (!response.ok || !response.body) {
    let detail = "An error occurred while connecting to the AI assistant.";
    try {
      detail = formatDetail((await response.json()).detail) || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop();
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (line) onEvent(JSON.parse(line.slice(5).trim()));
    }
  }
};
