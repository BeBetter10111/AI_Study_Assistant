# syntax=docker/dockerfile:1.7
# Multiple targets:
#   dev  (DEFAULT — final stage) : Vite dev server + hot reload, used with docker-compose.yml.
#   prod                         : static build served by non-root nginx, used with docker-compose.prod.yml.
#   docker build --target prod ./frontend

FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci

# ---- build: compile static assets into /app/dist -----------------------------
FROM deps AS build
# The API URL is baked into the bundle at build time. If empty, Vite reads VITE_API_BASE_URL from frontend/.env.local.
ARG VITE_API_BASE_URL=
COPY . .
RUN if [ -n "$VITE_API_BASE_URL" ]; then VITE_API_BASE_URL="$VITE_API_BASE_URL" npm run build; else npm run build; fi

# ---- prod: non-root nginx (nginx user, port 8080), static files only ----------
FROM nginxinc/nginx-unprivileged:1.27-alpine AS prod
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080

# ---- dev: MUST be the final stage so `build: ./frontend` defaults to the dev image ---
FROM deps AS dev
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]
