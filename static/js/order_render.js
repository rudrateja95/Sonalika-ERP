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

    <td class="text-center align-middle">${index + 1}</td>

    <td class="text-center align-middle">${order.order_no}</td>

    <td class="text-center align-middle">${order.client_code}</td>

    <td class="text-center align-middle">${order.company_name}</td>

    <td class="text-center align-middle">${order.total_qty}</td>

    <td class="text-center align-middle">${order.items.length}</td>

    <td class="text-center align-middle">
        <button class="btn btn-primary btn-sm"
            onclick="viewOrder('${order.order_no}', this)">
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