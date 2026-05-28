document.addEventListener('DOMContentLoaded', () => {
  // Theme toggle
  const themeBtn = document.getElementById('themeToggle');
  const themeIcon = document.getElementById('themeIcon');
  const applyTheme = (dark) => {
    document.body.classList.toggle('dark-mode', dark);
    themeIcon.className = dark ? 'bi bi-sun-fill' : 'bi bi-moon-stars-fill';
  };
  const saved = localStorage.getItem('pneumonia-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const isDark = saved ? saved === 'dark' : prefersDark;
  applyTheme(isDark);
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const nowDark = !document.body.classList.contains('dark-mode');
      localStorage.setItem('pneumonia-theme', nowDark ? 'dark' : 'light');
      applyTheme(nowDark);
    });
  }

  // Upload dropzone logic
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const previewContainer = document.getElementById('previewContainer');
  const previewImg = document.getElementById('previewImage');
  const uploadPrompt = document.getElementById('uploadPrompt');
  const removeBtn = document.getElementById('removePreviewBtn');
  if (dropzone) {
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      fileInput.files = e.dataTransfer.files;
      handleFile(file);
    });
    fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });
    if (removeBtn) removeBtn.addEventListener('click', () => { fileInput.value = ''; previewContainer.classList.add('d-none'); uploadPrompt.classList.remove('d-none'); });
  }
  function handleFile(file) {
    if (!file.type.startsWith('image/')) { alert('Please upload an image.'); return; }
    const reader = new FileReader();
    reader.onload = (e) => { previewImg.src = e.target.result; uploadPrompt.classList.add('d-none'); previewContainer.classList.remove('d-none'); };
    reader.readAsDataURL(file);
  }
});