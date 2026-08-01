function placeOrder() {

    console.log("===== PLACE ORDER START =====");

    var btn = document.getElementById("placeOrderBtn");
    var btnContent = document.getElementById("btnContent");

    btn.disabled = true;
    btnContent.innerHTML =
        '<span class="spinner-border spinner-border-sm me-2"></span>' +
        'Placing Order...';

    var items = [];
    var rows = document.querySelectorAll("#cartBody tr");

    for (var i = 0; i < rows.length; i++) {

        var row = rows[i];

        var item = {
            style_no: row.getAttribute("data-style-no"),
            qty: row.getAttribute("data-qty"),
            gold_color: row.getAttribute("data-gold-color"),
            gold_purity: row.getAttribute("data-gold-purity"),
            diamond_color: row.getAttribute("data-diamond-color"),
            diamond_clarity: row.getAttribute("data-diamond-clarity"),
            remarks: row.getAttribute("data-remarks")
        };

        console.log("Row " + (i + 1), item);
        items.push(item);
    }

    console.log("Total Items:", items.length);

    var payload = {
        items: items
    };

    console.log("Payload:", payload);

    fetch("/api/ecom-order", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    })
    .then(function (response) {

        console.log("Response Status:", response.status);

        return response.text();

    })
    .then(function (text) {

        console.log("Raw Response:", text);

        var data;

        try {
            data = JSON.parse(text);
        } catch (e) {

            btn.disabled = false;
            btnContent.innerHTML =
                '<i class="fas fa-check-circle"></i> Place Order';

            Swal.fire({
                icon: "error",
                title: "Invalid Response",
                text: "Server returned invalid JSON."
            });

            return;
        }

        console.log("Parsed Response:", data);

        if (data.ok) {

            Swal.fire({
                icon: "success",
                title: "Order Created",
                text: data.order_no,
                allowOutsideClick: false
            }).then(function () {
                location.reload();
            });

        } else {

            btn.disabled = false;
            btnContent.innerHTML =
                '<i class="fas fa-check-circle"></i> Place Order';

            Swal.fire({
                icon: "error",
                title: "Error",
                text: data.error
            });

        }

    })
    .catch(function (err) {

        console.error("Fetch Error:", err);

        btn.disabled = false;
        btnContent.innerHTML =
            '<i class="fas fa-check-circle"></i> Place Order';

        Swal.fire({
            icon: "error",
            title: "Exception",
            text: err.message
        });

    });

    console.log("===== PLACE ORDER END =====");
}