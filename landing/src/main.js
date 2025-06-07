
// Display user email on the success login page if available in localStorage or sessionStorage
document.addEventListener('DOMContentLoaded', () => {
    const emailSpan = document.getElementById('user-email');
    if (emailSpan) {
        // Try to get email from sessionStorage or localStorage
        const email = sessionStorage.getItem('user_email') || localStorage.getItem('user_email');
        if (email) {
            emailSpan.textContent = email;
        }
    }

    // Example: Save email after login (you should do this in your /auth backend route)
    // sessionStorage.setItem('user_email', user_email_from_backend);
});