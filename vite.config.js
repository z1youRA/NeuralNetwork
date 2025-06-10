import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import eslintPlugin from 'vite-plugin-eslint';
import glsl from 'vite-plugin-glsl';


// https://vitejs.dev/config/
export default defineConfig({
	plugins: [
		vue(),
		eslintPlugin({
			include: ['src/**/*.ts', 'src/**/*.vue', 'src/*.ts', 'src/*.vue'],
			overrideConfigFile: './eslint.config.cjs',
		}),
		glsl()
	],
	server: {
		host: '0.0.0.0', // 允许外部访问
		port: 5173, // 指定端口
	},
});
