try {
    # Получаем изменения
    git pull origin main
    
    # Добавляем все файлы
    git add .
    
    # Коммитим
    if (git status --porcelain) {
        git commit -m "Автоматическое обновление"
    }
    
    # Пушим
    git push origin main
    Write-Host "✅ Синхронизация успешна" -ForegroundColor Green
}
catch {
    Write-Host "❌ Ошибка: $_" -ForegroundColor Red
}