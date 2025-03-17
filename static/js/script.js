document.querySelector('form').addEventListener('submit', function(e) {
    let form = this;
    let ae = form.querySelector('[name="AE"]').value;
    let doc = form.querySelector('[name="DOC"]').value;
    let rmr = form.querySelector('[name="RMR"]').value;
    let gw = form.querySelector('[name="GW"]').value;
    let isValue = form.querySelector('[name="IS"]').value;

    if (!ae || !doc || !rmr || !gw || !isValue) {
        e.preventDefault();
        alert('Please fill in all fields');
    }
});