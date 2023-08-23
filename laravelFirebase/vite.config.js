import { defineConfig } from 'vite';
import laravel from 'laravel-vite-plugin';

export default defineConfig({
    build: {
        outDir: '../public',
    },
    plugins: [
        laravel({
            input: ['resources/css/app.css',],
            refresh: true,
        }),
    ],
});
