FROM node:22-alpine AS dependencies

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
RUN npm rebuild lightningcss

FROM dependencies AS builder

ARG NEXT_PUBLIC_LOCAL_API_URL=http://localhost:8000
ARG NEXT_PUBLIC_BACKEND_API_URL=http://localhost:8000/api
ENV NEXT_PUBLIC_LOCAL_API_URL=${NEXT_PUBLIC_LOCAL_API_URL}
ENV NEXT_PUBLIC_BACKEND_API_URL=${NEXT_PUBLIC_BACKEND_API_URL}

COPY src ./src
COPY public ./public
COPY next.config.mjs postcss.config.mjs tsconfig.json ./
RUN npm run build

FROM node:22-alpine AS runtime

WORKDIR /app
ENV NODE_ENV=production
COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.mjs ./next.config.mjs

EXPOSE 3000
CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0", "--port", "3000"]
