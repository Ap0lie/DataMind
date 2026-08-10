ARG DATAMIND_BUILD_SHA=local

FROM node:24-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY index.html postcss.config.js tailwind.config.js tsconfig.json tsconfig.node.json vite.config.ts ./
COPY src ./src

ARG VITE_DATAMIND_API_BASE_URL=/api/v1
ENV VITE_DATAMIND_API_BASE_URL=${VITE_DATAMIND_API_BASE_URL}

RUN npm run build

FROM nginx:1.27-alpine AS runtime

ARG DATAMIND_BUILD_SHA
LABEL org.opencontainers.image.revision=${DATAMIND_BUILD_SHA}

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=5 \
  CMD wget -q -O /dev/null http://127.0.0.1/healthz || exit 1
