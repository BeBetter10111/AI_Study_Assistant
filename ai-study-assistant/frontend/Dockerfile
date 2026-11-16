# syntax=docker/dockerfile:1.7
# Nhiều "target":
#   dev  (MẶC ĐỊNH — stage cuối)  : Vite dev server + hot reload, dùng với docker-compose.yml.
#   prod                           : bản build tĩnh phục vụ bằng nginx KHÔNG-root, dùng với docker-compose.prod.yml.
#   docker build --target prod ./frontend

FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci

# ---- build: biên dịch ra file tĩnh trong /app/dist ---------------------------
FROM deps AS build
# URL API được "đóng" vào bundle lúc build. Để trống => Vite đọc VITE_API_BASE_URL trong frontend/.env.local.
ARG VITE_API_BASE_URL=
COPY . .
RUN if [ -n "$VITE_API_BASE_URL" ]; then VITE_API_BASE_URL="$VITE_API_BASE_URL" npm run build; else npm run build; fi

# ---- prod: nginx không-root (user nginx, cổng 8080), chỉ chứa file tĩnh -------
FROM nginxinc/nginx-unprivileged:1.27-alpine AS prod
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080

# ---- dev: PHẢI là stage cuối để `build: ./frontend` mặc định vẫn ra bản dev ---
FROM deps AS dev
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]
