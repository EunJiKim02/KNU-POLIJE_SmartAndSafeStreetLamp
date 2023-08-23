/ @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./resources/**/*.blade.php",
        "./resources/**/*.js",
        "./resources/**/*.vue",
    ],
    theme: {
        extend: {
            fontFamily: {
                'poppins-regular': 'poppins-regular',
                'poppins-semibold': 'poppins-semibold',
                'poppins-bold': 'poppins-bold',
                'poppins-medium': 'poppins-medium'
              },
        },
    },
    plugins: [],
};