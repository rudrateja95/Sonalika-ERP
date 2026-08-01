async function placeOrder() {

    console.log("===== PLACE ORDER START =====");

    const btn = document.getElementById("placeOrderBtn");
    const btnContent = document.getElementById("btnContent");

    // Show Loader
    btn.disabled = true;
    btnContent.innerHTML = `
        <span class="spinner-border spinner-border-sm me-2"></span>
        Placing Order...
    `;

    try {

        const items = [];

        document.querySelectorAll("#cartBody tr").forEach((row, index) => {

            const item = {

                style_no: row.dataset.styleNo,
                qty: row.dataset.qty,
                gold_color: row.dataset.goldColor,
                gold_purity: row.dataset.goldPurity,
                diamond_color: row.dataset.diamondColor,
                diamond_clarity: row.dataset.diamondClarity,
                remarks: row.dataset.remarks

            };

            console.log("Row", index + 1, item);

            items.push(item);

        });

        console.log("Total Items:", items.length);

        const payload = {
            items: items
        };

        console.log("Payload:", payload);

        const response = await fetch("/api/ecom-order", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)

        });

        console.log("Response Status:", response.status);

        const text = await response.text();

        console.log("Raw Response:", text);

        let data;

        try {

            data = JSON.parse(text);

        } catch (e) {

            btn.disabled = false;

            btnContent.innerHTML = `
                <i class="fas fa-check-circle"></i>
                Place Order
            `;

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
            }).then(() => {

                location.reload();

            });

        } else {

            btn.disabled = false;

            btnContent.innerHTML = `
                <i class="fas fa-check-circle"></i>
                Place Order
            `;

            Swal.fire({
                icon: "error",
                title: "Error",
                text: data.error
            });

        }

    } catch (err) {

        console.error("Fetch Error:", err);

        btn.disabled = false;

        btnContent.innerHTML = `
            <i class="fas fa-check-circle"></i>
            Place Order
        `;

        Swal.fire({
            icon: "error",
            title: "Exception",
            text: err.message
        });

    }

    console.log("===== PLACE ORDER END =====");

}