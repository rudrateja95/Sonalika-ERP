async function placeOrder() {

    console.log("===== PLACE ORDER START =====");

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

        console.log("Sending request...");

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

            console.error("JSON Parse Error:", e);

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
                text: data.order_no
            });

        } else {

            Swal.fire({
                icon: "error",
                title: "Error",
                text: data.error
            });

        }

    } catch (err) {

        console.error("Fetch Error:", err);

        Swal.fire({
            icon: "error",
            title: "Exception",
            text: err.message
        });

    }

    console.log("===== PLACE ORDER END =====");
}
