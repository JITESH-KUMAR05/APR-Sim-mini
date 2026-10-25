/**
 * Theme state for the APR console: dark or light. A first visit follows the operating system,
 * the choice is remembered. Drawing code asks APRTheme.color('accent') for a token's value and
 * listens for 'apr-theme-change' to redraw.
 */
(function () {
    const KEY = 'apr-theme';
    const root = document.documentElement;

    function stored() {
        try { return localStorage.getItem(KEY); } catch (e) { return null; }
    }

    function remember(value) {
        try { localStorage.setItem(KEY, value); } catch (e) { /* storage blocked: still switch */ }
    }

    function preferred() {
        const saved = stored();
        if (saved === 'dark' || saved === 'light') return saved;
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }

    function apply(value) {
        root.setAttribute('data-theme', value);
        document.dispatchEvent(new CustomEvent('apr-theme-change', { detail: { theme: value } }));
    }

    function color(name) {
        return getComputedStyle(root).getPropertyValue('--' + name).trim();
    }

    apply(preferred());

    window.APRTheme = {
        get: () => root.getAttribute('data-theme'),
        set: (value) => { remember(value); apply(value); },
        toggle: () => window.APRTheme.set(window.APRTheme.get() === 'dark' ? 'light' : 'dark'),
        color,
    };
})();
