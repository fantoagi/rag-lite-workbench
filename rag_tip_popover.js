/**
 * 感叹号提示：checkbox+label 控制显示；气泡 position:fixed，在 label（!）右侧，不够放则翻到左侧。
 * 由 main.py 通过 demo.launch(head=...) 注入，不依赖 Blocks(js=)。
 */
(function () {
  'use strict';

  var GAP = 8;

  function positionPop(pop, trigger) {
    var r = trigger.getBoundingClientRect();
    var vw = window.innerWidth;
    var vh = window.innerHeight;
    var pw = pop.offsetWidth || 360;
    var ph = pop.offsetHeight || 100;

    var left = r.right + GAP;
    var top = r.top + (r.height - ph) / 2;

    if (left + pw > vw - 8) {
      left = r.left - GAP - pw;
    }
    if (left < 8) {
      left = 8;
    }
    if (left + pw > vw - 8) {
      left = Math.max(8, vw - 8 - pw);
    }

    if (top + ph > vh - 8) {
      top = vh - 8 - ph;
    }
    if (top < 8) {
      top = 8;
    }

    pop.style.position = 'fixed';
    pop.style.left = left + 'px';
    pop.style.top = top + 'px';
    pop.style.zIndex = '2147483000';
    pop.style.right = 'auto';
    pop.style.bottom = 'auto';
    pop.style.margin = '0';
    pop.style.transform = 'none';
  }

  function bindAnchor(anchor) {
    if (anchor.dataset.ragTipBound) return;
    var cb = anchor.querySelector('.rag-tip-cb');
    var pop = anchor.querySelector('.rag-tip-pop');
    var trigger = anchor.querySelector('label.rag-tip-trigger');
    if (!cb || !pop || !trigger) return;
    anchor.dataset.ragTipBound = '1';

    cb.addEventListener('change', function () {
      if (!cb.checked) return;
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          positionPop(pop, trigger);
        });
      });
    });
  }

  function scan() {
    document.querySelectorAll('.rag-tip-anchor').forEach(bindAnchor);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', scan);
  } else {
    scan();
  }

  function scrollBuildLogsToBottom() {
    document.querySelectorAll('#rag_kb_build_log textarea, .rag-kb-build-log textarea').forEach(function (ta) {
      try {
        ta.scrollTop = ta.scrollHeight;
      } catch (e) {}
    });
  }

  var debounce;
  new MutationObserver(function () {
    clearTimeout(debounce);
    debounce = setTimeout(function () {
      scan();
      scrollBuildLogsToBottom();
    }, 120);
  }).observe(document.body, { childList: true, subtree: true, characterData: true });
})();
