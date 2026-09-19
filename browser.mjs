// Сквозная проверка страницы в настоящем браузере.
//
// Зачем отдельно от tests.py: тесты на Python проверяют сервер и структуру
// скрипта, но не могут выполнить страницу. Ошибка, ради которой этот прогон и
// появился, была именно такой — «запуститьСценарий» оказался объявлен внутри
// другой функции. Синтаксис корректен, сервер отвечает, все 150 тестов зелёные,
// а кнопка «Спросить» падает с ReferenceError, и сценарий не запускается вовсе.
//
// Как запускать:
//   1) поднять сервер:   MEMORY_DIR=/tmp/проба PORT=5000 python web.py
//   2) поднять браузер:  google-chrome --headless=new --remote-debugging-port=9222 \
//                          --no-sandbox --disable-gpu about:blank
//   3) сам прогон:       node browser.mjs [адрес] [ключ-модели] [запрос] [режим]
//
// Режимы:
//   сценарий (по умолчанию) — запустить сценарий по триггеру и дождаться конца;
//   пауза                   — включить «подтверждать смену стадии», дождаться
//                             остановки, нажать «Продолжить» и убедиться, что
//                             задача сдвинулась;
//   переходы                — завести задачу, нажать запертую кнопку перехода и
//                             убедиться, что страница показывает разбор отказа,
//                             а после плана и утверждения переход открывается.
//
// По умолчанию: http://127.0.0.1:5000, модель ds-flash. Прогон печатает, что
// появляется в чате, и — главное — исключения JS, которых в чате не видно.

const АДРЕС = process.argv[2] || 'http://127.0.0.1:5000';
const МОДЕЛЬ = process.argv[3] || 'ds-flash';
const РЕЖИМ = process.argv[5] || 'сценарий';
const ЗАПРОС = process.argv[4] ||
  'Напиши фичу по получению данных о кадастровых участках с ресурса ' +
  'https://nspd.gov.ru/map и вывода их отдельным слоем на карте ГИС с газопроводами.';
const CDP = process.env.CDP || 'http://127.0.0.1:9222';
const ПРЕДЕЛ_МС = 300000;

const цель = await (await fetch(`${CDP}/json/new?${encodeURIComponent(АДРЕС)}`,
                                {method: 'PUT'})).json();
const ws = new WebSocket(цель.webSocketDebuggerUrl);
const ждущие = new Map();
const ошибки = [];
let счётчик = 0;

ws.addEventListener('message', (событие) => {
  const м = JSON.parse(событие.data);
  if (м.id && ждущие.has(м.id)) { ждущие.get(м.id)(м); ждущие.delete(м.id); }
  if (м.method === 'Runtime.exceptionThrown') {
    const д = м.params.exceptionDetails;
    ошибки.push('исключение: ' + (д.exception?.description || д.text));
  }
  if (м.method === 'Runtime.consoleAPICalled' && м.params.type === 'error') {
    ошибки.push('console.error: ' +
                м.params.args.map(а => а.value ?? а.description).join(' '));
  }
});
await new Promise(р => ws.addEventListener('open', р));

const зов = (метод, параметры = {}) => new Promise(р => {
  const id = ++счётчик;
  ждущие.set(id, р);
  ws.send(JSON.stringify({id, method: метод, params: параметры}));
});

const выполнить = async (код) => {
  const о = await зов('Runtime.evaluate',
                      {expression: код, awaitPromise: true, returnByValue: true});
  if (о.result?.exceptionDetails) ошибки.push('evaluate: ' + о.result.exceptionDetails.text);
  return о.result?.result?.value;
};

await зов('Runtime.enable');
await зов('Log.enable');
await new Promise(р => setTimeout(р, 3000));    // страница подтягивает состояние

console.log('страница:', await выполнить('document.title'));
console.log('модель:', await выполнить(
  `(() => { const с = document.getElementById('модель');
            с.value = ${JSON.stringify(МОДЕЛЬ)};
            с.dispatchEvent(new Event('change'));
            return с.value; })()`));

if (РЕЖИМ === 'пауза') {
  console.log('режим «по шагам»:', await выполнить(
    `(() => { const г = document.getElementById('режим-по-шагам');
              г.checked = true; г.dispatchEvent(new Event('change'));
              return г.checked; })()`));
}

if (РЕЖИМ === 'переходы') {
  // Отдельный ход: здесь проверяется жизненный цикл задачи, а не сценарий.
  // Модель зовётся один раз — на составление плана, без которого нечего
  // утверждать, а значит, и нечего открывать.
  const имя = 'ворота-' + Date.now().toString().slice(-6);
  await выполнить(`(() => {
    document.getElementById('новая-задача').value = ${JSON.stringify(имя)};
    document.getElementById('название').value = 'проверка ворот из браузера';
    document.getElementById('создать').click(); return 'ок'; })()`);
  await new Promise(р => setTimeout(р, 2500));

  const кнопки = await выполнить(`Array.from(
    document.querySelectorAll('#переходы button')).map(к => к.textContent.trim())`);
  console.log('кнопки переходов:', кнопки);
  const запертых = (кнопки || []).filter(т => т.includes('🔒')).length;
  console.log('запертых кнопок:', запертых);
  if (!кнопки || кнопки.length < 3) ошибки.push('на странице нет кнопок переходов');
  if (!запертых) ошибки.push('все переходы показаны открытыми — ворота не видны');

  // Нажимаем запертый переход: страница должна показать разбор отказа.
  await выполнить(`(() => {
    const к = Array.from(document.querySelectorAll('#переходы button'))
      .find(к => к.textContent.includes('🔒'));
    if (к) к.click(); return к ? к.textContent : 'нет'; })()`);
  await new Promise(р => setTimeout(р, 2000));
  const отказов = await выполнить(`document.querySelectorAll('#чат .отказ').length`);
  const текстОтказа = await выполнить(
    `(document.querySelector('#чат .отказ') || {}).innerText || ''`);
  console.log('отказов в чате:', отказов);
  console.log('первые строки отказа:\n' + (текстОтказа || '(пусто)').split('\n').slice(0, 6).join('\n'));
  if (!отказов) ошибки.push('нажатие запертого перехода не дало отказа');

  console.log('\nсоставляю план (один вызов модели)…');
  await выполнить(`document.getElementById('план').click(); 'ок'`);
  const началоПлана = Date.now();
  while (Date.now() - началоПлана < 180000) {
    await new Promise(р => setTimeout(р, 3000));
    const занято = await выполнить(`document.getElementById('отправить').disabled`);
    if (!занято && Date.now() - началоПлана > 6000) break;
  }
  const доУтверждения = await выполнить(`document.getElementById('ворота').innerText`);
  console.log('\n=== ворота до утверждения ===\n' + (доУтверждения || '(пусто)'));

  await выполнить(`document.getElementById('утвердить-план').click(); 'ок'`);
  await new Promise(р => setTimeout(р, 2500));
  const послеУтверждения = await выполнить(`document.getElementById('ворота').innerText`);
  console.log('\n=== ворота после утверждения ===\n' + (послеУтверждения || '(пусто)'));

  const открыт = await выполнить(`(() => {
    const к = Array.from(document.querySelectorAll('#переходы button'))
      .find(к => к.textContent.includes('исполнение'));
    return к ? к.textContent.trim() : 'нет кнопки'; })()`);
  console.log('кнопка исполнения после утверждения:', открыт);
  if (открыт.includes('🔒')) ошибки.push('после утверждения плана переход остался закрытым');

  const сбоиПереходов = await выполнить(`document.querySelectorAll('#чат .ошибка').length`);
  console.log('\n=== итог ===');
  console.log('сообщений об ошибке в чате:', сбоиПереходов);
  console.log('исключений JS:', ошибки.length ? ошибки : 'нет');
  ws.close();
  process.exit(ошибки.length ? 1 : 0);
}

await выполнить(`document.getElementById('ввод').value = ${JSON.stringify(ЗАПРОС)};
                 document.getElementById('отправить').click(); 'пуск'`);

const начало = Date.now();
let прошлое = '';
while (Date.now() - начало < ПРЕДЕЛ_МС) {
  await new Promise(р => setTimeout(р, 3000));
  const чат = await выполнить(`document.getElementById('чат').innerText`);
  const новое = (чат || '').slice(прошлое.length);
  if (новое.trim()) {
    for (const строка of новое.split('\n').filter(с => с.trim()).slice(0, 3)) {
      console.log(`[${((Date.now() - начало) / 1000).toFixed(0)} с] ${строка.slice(0, 110)}`);
    }
    прошлое = чат;
  }
  const занято = await выполнить(`document.getElementById('отправить').disabled`);
  if (!занято && Date.now() - начало > 8000) break;
}

if (РЕЖИМ === 'пауза') {
  const состояние = await выполнить(
    `document.getElementById('состояние-задачи').innerText`);
  console.log('\n=== состояние задачи на паузе ===\n' + (состояние || '(пусто)'));
  const наПаузе = (состояние || '').includes('на паузе');
  console.log('карточка показывает паузу:', наПаузе ? 'да' : 'НЕТ');

  console.log('\nжму «Продолжить»…');
  await выполнить(`document.getElementById('продолжить-задачу').click(); 'ок'`);
  const начало2 = Date.now();
  while (Date.now() - начало2 < 120000) {
    await new Promise(р => setTimeout(р, 3000));
    const занято = await выполнить(`document.getElementById('отправить').disabled`);
    if (!занято && Date.now() - начало2 > 6000) break;
  }
  const после = await выполнить(`document.getElementById('состояние-задачи').innerText`);
  console.log('\n=== состояние после «Продолжить» ===\n' + (после || '(пусто)'));
  console.log('состояние сдвинулось:', после !== состояние ? 'да' : 'НЕТ');
  if (!наПаузе || после === состояние) ошибки.push('пауза или продолжение не сработали');
}

const чат = await выполнить(`document.getElementById('чат').innerText`);
const отвечено = await выполнить(`document.querySelectorAll('#чат .от-агента').length`);
const сбои = await выполнить(`document.querySelectorAll('#чат .ошибка').length`);

console.log('\n=== итог ===');
console.log('символов в чате:', (чат || '').length);
console.log('ответов агента:', отвечено);
console.log('сообщений об ошибке в чате:', сбои);
console.log('исключений JS:', ошибки.length ? ошибки : 'нет');

ws.close();
// Ненулевой код — чтобы прогон годился и для проверки перед сдачей.
process.exit(ошибки.length || сбои || !отвечено ? 1 : 0);
