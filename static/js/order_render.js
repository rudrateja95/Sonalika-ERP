async function loadOrders() {

    try {

        const res = await fetch("/api/orders");

        if (!res.ok) {
            throw new Error("HTTP " + res.status);
        }

        const orders = await res.json();

        let html = "";

        if (orders.length === 0) {
            html = `
                <tr>
                    <td colspan="6" class="text-center">
                        No Orders Found
                    </td>
                </tr>
            `;
        } else {

            orders.forEach((order, index) => {

                html += `
                    <tr>
                        <td>${index + 1}</td>
                        <td>${order.order_no}</td>
                        <td>${order.client_id}</td>
                        <td>${order.total_qty}</td>
                        <td>${order.items.length}</td>
                        <td>
                            <button class="btn btn-primary btn-sm"
                                onclick="viewOrder('${order.order_no}')">
                                View
                            </button>
                        </td>
                    </tr>
                `;

            });

        }

        document.getElementById("orderTable").innerHTML = html;

    } catch (err) {

        console.error(err);

        document.getElementById("orderTable").innerHTML = `
            <tr>
                <td colspan="6" class="text-danger text-center">
                    Failed to load orders.
                </td>
            </tr>
        `;
    }
}

loadOrders();