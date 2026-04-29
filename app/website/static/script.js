const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('imageInput');
const messageDiv = document.getElementById('message');
const previewDiv = document.getElementById('preview');

fileInput.addEventListener('change', async (e) => {
    const file = fileInput.files[0];
    if (!file) {
        showMessage('Выберите файл', 'error',messageDiv);
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            previewDiv.innerHTML = `<img src="${data.url}" alt="Загруженное изображение">`;
        } else {
            showMessage(data.error || 'Ошибка загрузки', 'error',messageDiv);
        }
    } catch (err) {
        showMessage('Сетевая ошибка: ' + err.message, 'error',messageDiv);
    }
});

function showMessage(text, type, messageContainer) {
    messageContainer.textContent = text;
    messageContainer.className = type === 'error' ? 'error' : '';
}

const determineForm = document.getElementById('determineForm');
const resultHeading = document.querySelector('.predicted_text h1');

determineForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    try {
        const response = await fetch('/determine', {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            resultHeading.textContent = data.result;
        } else {
            resultHeading.textContent = 'Ошибка: ' + (data.error || 'неизвестная ошибка');
        }
    } catch (err) {
        resultHeading.textContent = 'Сетевая ошибка: ' + err.message;
    }
});

const uploadForm = document.getElementById('uploadFolder');
const uploadMessage = document.getElementById('uploadMessage');

async function uploadFolder(files) {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {

        const file = files[i];
        const relativePath = files[i].webkitRelativePath || file.name;
        
        formData.append('images', file, relativePath.replace(/\\/g, '/'));
    }

    try {
        const response = await fetch('/server/upload', {
            method: 'POST',
            body: formData
        })
        const data = await response.json();
        let classStats = '';
        if (data.class && typeof data.class === 'object') {
            const entries = Object.entries(data.class);
            if (entries.length > 0) {
                classStats = entries.map(([cls, count]) => `${cls}: ${count}`).join(', ');
            } else {
                classStats = 'нет';
            }
        } else {
            classStats = 'неизвестно';
        }
        if (response.ok) {
            showMessage(`${data.saved} файлов загружено классы: ${classStats},
                пропущено ${data.duplicates === null || data.duplicates === undefined ? 0 : data.duplicates} файлов(дупликация)`, 
                'success',uploadMessage);
        }
        else {
            showMessage(data.error || 'Ошибка загрузки', 'error',uploadMessage);
        }
    }
    catch (err) {
        showMessage('Сетевая ошибка: ' + err.message, 'error',uploadMessage);
    }
}

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const files = document.getElementById('folderInput').files;
    if (!files.length) {
        showMessage('Выберите файлы', 'error',uploadMessage);
        return;
    }
    await uploadFolder(files);
})

const modelInfoDiv = document.getElementById('modelInfo');
const eventSource = new EventSource('/stream');

eventSource.onmessage = function(event) {
    modelInfoDiv.innerHTML = `Информация о модели: ${event.data}`;
};

eventSource.onerror = function(event) {
    console.error('SSE error:', event);
    modelInfoDiv.innerHTML = 'Ошибка соединения с сервером. Переподключение...';
};
